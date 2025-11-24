import os
import sys
import glob
import logging
import torch
import numpy as np
import transforms3d
import plotly.graph_objects as go
import plotly.io as pio
from omegaconf import DictConfig, OmegaConf
import hydra
import re
import imageio
import trimesh as tm
import pyrender
import multiprocessing
import trimesh
from pathlib import Path

from mr_utils.utils_calc import posQuat2Isometry3d, quatWXYZ2XYZW

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.config import ExperimentConfig
from utils.common import setup_device, set_random_seeds, ensure_directory
from utils.dual_arm_hand_model import DualArmHandModel

translation_names = ["WRJTx", "WRJTy", "WRJTz"]
rot_names = ["WRJRx", "WRJRy", "WRJRz"]


def natural_key(filename):
    """Extract numeric part from the filename for natural sorting."""
    match = re.search(r"_(\d+)\.npy$", filename)
    return int(match.group(1)) if match else float("inf")


def experiment_config_from_dict(cfg: DictConfig) -> ExperimentConfig:
    """Convert a Hydra DictConfig to ExperimentConfig dataclass."""
    exp = ExperimentConfig()

    # Top level simple fields
    for key in ("name", "seed", "gpu"):
        if key in cfg:
            setattr(exp, key, cfg.get(key))

    # Helper to apply nested dict to dataclass-like object
    def apply_section(section_name, target_obj):
        if section_name in cfg:
            sec = cfg.get(section_name)
            for k, v in sec.items():
                if hasattr(target_obj, k):
                    # convert lists/dicts to native Python
                    val = OmegaConf.to_object(v) if isinstance(v, (dict, list)) else v
                    setattr(target_obj, k, val)

    apply_section("hand_params", exp.hand_params)
    apply_section("paths", exp.paths)
    apply_section("energy", exp.energy)
    apply_section("optimizer", exp.optimizer)
    apply_section("initialization", exp.initialization)
    apply_section("model", exp.model)

    return exp


def get_scene(plot_lst):
    """
    plotly utils
    """
    xs, ys, zs = [], [], []
    for trace in plot_lst:
        if hasattr(trace, "x") and hasattr(trace, "y") and hasattr(trace, "z"):
            xs.extend(trace.x)
            ys.extend(trace.y)
            zs.extend(trace.z)

    xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    zmin, zmax = zs.min(), zs.max()

    # 计算各轴范围
    x_range = xmax - xmin
    y_range = ymax - ymin
    z_range = zmax - zmin

    # 计算比例（归一化到最长轴为1）
    max_range = max(x_range, y_range, z_range)
    aspectratio = dict(
        x=x_range / max_range,
        y=y_range / max_range,
        z=z_range / max_range,
    )

    scene_fixed = dict(
        xaxis=dict(range=[xmin, xmax], visible=False, showgrid=False),
        yaxis=dict(range=[ymin, ymax], visible=False, showgrid=False),
        zaxis=dict(range=[zmin, zmax], visible=False, showgrid=False),
        aspectmode="manual",
        aspectratio=aspectratio,
    )
    return scene_fixed


def look_at(cam_pos, target=np.array([0, 0, 0]), up=np.array([0, 0, 1])):
    """
    Compute a 4x4 camera pose matrix that points from cam_pos to target.

    Args:
        cam_pos: (3,) array-like, camera position in world coordinates
        target: (3,) array-like, point to look at
        up: (3,) array-like, world up direction

    Returns:
        pose: (4,4) numpy array, camera pose matrix for pyrender
    """
    cam_pos = np.array(cam_pos, dtype=np.float64)
    target = np.array(target, dtype=np.float64)
    up = np.array(up, dtype=np.float64)

    # forward vector (from camera to target)
    forward = target - cam_pos
    forward /= np.linalg.norm(forward)

    # right vector
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    # true up vector
    true_up = np.cross(right, forward)
    true_up /= np.linalg.norm(true_up)

    # construct 4x4 pose matrix
    pose = np.eye(4)
    pose[:3, 0] = right  # X axis
    pose[:3, 1] = true_up  # Y axis
    pose[:3, 2] = -forward  # Z axis points forward in pyrender
    pose[:3, 3] = cam_pos  # translation

    return pose


# --- Utility to build hand pose tensor ---
def build_hand_pose(qpos, translation_names, rot_names, joint_names, device):
    """Build a torch tensor for hand pose given qpos dict."""
    rot = np.array(transforms3d.euler.euler2mat(*[qpos[name] for name in rot_names]))
    rot = rot[:, :2].T.ravel().tolist()  # flatten first two rotation columns
    hand_pose = torch.tensor(
        [qpos[name] for name in translation_names] + rot + [qpos[name] for name in joint_names],
        dtype=torch.float,
        device=device,
    )
    return hand_pose


def build_qpos(qpos_dict, joint_names, device):
    """Build a torch tensor for hand pose given qpos dict."""
    qpos = torch.tensor(
        [qpos_dict[name] for name in joint_names],
        dtype=torch.float,
        device=device,
    )
    return qpos


def create_colored_axes(origin_size=0.01, axis_length=0.1, radius=0.002):
    """
    Create separate trimesh meshes: origin sphere, x-axis (red), y-axis (green), z-axis (blue).
    Correctly rotate cylinders (默认 cylinder 沿 z 轴) then translate along target axis.
    Returns: [origin_sphere, x_axis_mesh, y_axis_mesh, z_axis_mesh]
    """
    from trimesh.transformations import rotation_matrix

    meshes = []

    # 原点小球
    origin = trimesh.creation.icosphere(subdivisions=2, radius=origin_size)
    origin.visual.vertex_colors = np.tile(np.array([255, 255, 255, 255], dtype=np.uint8), (len(origin.vertices), 1))
    meshes.append(origin)

    # 基本圆柱（沿 z 轴，center 在原点，高度 = axis_length）
    cyl_z = trimesh.creation.cylinder(radius=radius, height=axis_length, sections=32)
    # Note: cylinder is centered on origin (extends from -h/2 to +h/2 along local z)

    # --- Z 轴（蓝），无需旋转，只需平移 +h/2 到正方向 ---
    z_axis = cyl_z.copy()
    z_axis.apply_translation([0.0, 0.0, axis_length / 2.0])
    z_axis.visual.vertex_colors = np.tile(np.array([0, 0, 255, 255], dtype=np.uint8), (len(z_axis.vertices), 1))
    meshes.append(z_axis)

    # --- X 轴（红）: 把 z 映到 x，绕 y 轴旋转 +90deg (pi/2) ---
    x_axis = cyl_z.copy()
    R_x = rotation_matrix(np.pi / 2.0, [0, 1, 0])  # 旋转矩阵 4x4
    x_axis.apply_transform(R_x)  # 先旋转
    # 旋转后，局部 z 对齐到世界 x，因此平移沿世界 x
    x_axis.apply_translation([axis_length / 2.0, 0.0, 0.0])
    x_axis.visual.vertex_colors = np.tile(np.array([255, 0, 0, 255], dtype=np.uint8), (len(x_axis.vertices), 1))
    meshes.append(x_axis)

    # --- Y 轴（绿）: 把 z 映到 y，绕 x 轴旋转 -90deg (-pi/2) ---
    y_axis = cyl_z.copy()
    R_y = rotation_matrix(-np.pi / 2.0, [1, 0, 0])
    y_axis.apply_transform(R_y)
    # 旋转后平移沿世界 y
    y_axis.apply_translation([0.0, axis_length / 2.0, 0.0])
    y_axis.visual.vertex_colors = np.tile(np.array([0, 255, 0, 255], dtype=np.uint8), (len(y_axis.vertices), 1))
    meshes.append(y_axis)

    return meshes


def save_grasp_images(params):
    file_path, cfg = params[0], params[1]

    if not os.path.exists(file_path):
        logging.warning(f"File {file_path} does not exist.")
        return

    # merge cfg.hand.paths into cfg.paths
    cfg.paths = OmegaConf.merge(cfg.paths, cfg.hand.paths)
    cfg.hand_params = OmegaConf.merge(cfg.hand_params, cfg.hand.hand_params)
    cfg.initialization = OmegaConf.merge(cfg.initialization, cfg.hand.initialization)
    # Convert to ExperimentConfig dataclass
    # config = experiment_config_from_dict(cfg)

    exp_path = os.path.join(cfg.paths.experiments_base, cfg.name)
    device = setup_device(cfg.gpu)

    path_obj = Path(file_path)
    object_code = path_obj.parent.name
    filename = path_obj.stem

    dual_arm_hand_model = DualArmHandModel(
        n_surface_points=cfg.model.n_surface_points,
        device=device,
        cfg=cfg.dual_arm_hand,
    )
    joint_names = dual_arm_hand_model.joints_names

    object_model = ObjectModel(
        data_root_path=cfg.paths.data_root_path,
        batch_size_each=1,
        num_samples=2000,
        device=device,
        size="large",
        bodex_format=True,
    )
    object_model.initialize([object_code])

    ####################### Load data #######################

    data_dict = np.load(file_path, allow_pickle=True).item()
    grasp_qpos_dict = data_dict["dual_arm_hand"]["grasp_qpos"]
    obj_scale = data_dict["scale"]
    obj_pose7d = data_dict["dual_arm_hand"]["obj_pose"]
    obj_pose = posQuat2Isometry3d(obj_pose7d[:3], quatWXYZ2XYZW(obj_pose7d[3:]))

    object_model.object_scale_tensor[0] = obj_scale
    object_model.set_parameters(poses=torch.tensor(obj_pose, device=device).unsqueeze(0))

    grasp_qpos = build_qpos(grasp_qpos_dict, joint_names, device=device)
    dual_arm_hand_model.set_parameters(grasp_qpos.unsqueeze(0))

    ####################### Rendering #######################

    robot_mesh = dual_arm_hand_model.get_trimesh_data(0, rgba=[0.467, 0.533, 0.600, 0.7])
    object_mesh = object_model.get_trimesh_data(i=0, rgba=[1.0, 0.498, 0.314, 0.5])
    axis_mesh = create_colored_axes(origin_size=0.005, axis_length=1.0, radius=0.002)

    all_meshes = robot_mesh + [object_mesh] + axis_mesh

    # 创建 pyrender 场景
    scene = pyrender.Scene(bg_color=[226 / 255, 240 / 255, 217 / 255, 1.0])  # 背景色 #E2F0D9\

    # 添加所有 mesh
    for m in all_meshes:
        # 使用默认材质，支持 RGBA
        if hasattr(m.visual, "vertex_colors") and m.visual.vertex_colors.shape[1] == 4:
            # 如果 Trimesh 有 RGBA
            color = m.visual.vertex_colors[0] / 255.0
        else:
            color = [0.8, 0.8, 0.8, 1.0]
        material = pyrender.MetallicRoughnessMaterial(baseColorFactor=color, metallicFactor=0.0, roughnessFactor=0.9)
        mesh_pyr = pyrender.Mesh.from_trimesh(m, material=material)
        scene.add(mesh_pyr)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)

    os.environ["PYOPENGL_PLATFORM"] = "egl"  # use EGL on headless server
    r = pyrender.OffscreenRenderer(viewport_width=1920, viewport_height=1080)

    save_dir = os.path.join(exp_path, cfg.task.save_dir, f"{object_code}")
    os.makedirs(save_dir, exist_ok=True)

    cam_pose_lst = cfg.task.camera.pos_lst
    cam_center = cfg.task.camera.center
    for i_cam, cam_pose in enumerate(cam_pose_lst):
        cam_pose = look_at(cam_pose, cam_center)
        cam_node = scene.add(camera, pose=cam_pose)
        light_node = scene.add(light, pose=cam_pose)
        color, _ = r.render(scene)
        scene.remove_node(cam_node)  # 移除相机
        scene.remove_node(light_node)  # 移除光源
        path = os.path.join(save_dir, f"{filename}_cam_{i_cam}.jpg")
        imageio.imwrite(path, color)

    r.delete()  # 释放 GPU/CPU 资源
    print(f"Saved images: {save_dir}/{filename}")


def task_arm_visualize(cfg: DictConfig):
    exp_path = os.path.join(cfg.paths.experiments_base, cfg.name)
    source_path = os.path.join(exp_path, cfg.task.source_dir)
    all_grasp_file_lst = glob.glob(os.path.join(source_path, "**/*.npy"), recursive=True)

    # Select grasp data files with the specified object codes and grasp idx
    object_code_list = list(cfg.task.object_code_list)
    grasp_idx_lst = list(cfg.task.grasp_idx_lst)
    selected_grasp_file_lst = []
    for grasp_file in all_grasp_file_lst:
        # 提取物体ID
        path_obj = Path(grasp_file)
        object_code = path_obj.parent.name  # core_bottle_fa452996a220dd49257c84a0e95a18b0
        filename = path_obj.stem  # grasp_3_pose_1
        grasp_match = re.search(r"grasp_(\d+)", filename)
        if grasp_match:
            grasp_idx = int(grasp_match.group(1)) if grasp_match else None  # 3
        else:
            raise ValueError("Invalid filename format!")

        if (object_code in object_code_list) and (grasp_idx in grasp_idx_lst):
            selected_grasp_file_lst.append(grasp_file)

    logging.info(f"Select {len(selected_grasp_file_lst)}/{len(all_grasp_file_lst)} grasp file.")

    # Pass the 'vis_start_pose' param into the cfg
    cfg_lst = [cfg] * len(selected_grasp_file_lst)
    iterable_params = zip(selected_grasp_file_lst, cfg_lst)

    if cfg.task.debug:
        for params in iterable_params:
            save_grasp_images(params)
    else:
        with multiprocessing.Pool(processes=cfg.n_worker) as pool:
            result_iter = pool.imap_unordered(save_grasp_images, iterable_params)
            results = list(result_iter)

    save_dir = os.path.join(exp_path, cfg.task.save_dir)
    img_lst = glob.glob(os.path.join(save_dir, "**/*.jpg"), recursive=True)
    logging.info(f"Save {len(img_lst)} images in {save_dir}.")
    logging.info("Finish grasp image saving.")

    return
