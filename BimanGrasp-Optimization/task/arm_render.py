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
from utils.object_model import ObjectModel
from utils.common import setup_device
from utils.dual_arm_hand_model import DualArmHandModel
from utils.visualization import look_at, create_colored_axes


def build_qpos(qpos_dict, joint_names, device):
    """Build a torch tensor for hand pose given qpos dict."""
    qpos = torch.tensor(
        [qpos_dict[name] for name in joint_names],
        dtype=torch.float,
        device=device,
    )
    return qpos


def save_grasp_images(params):
    file_path, cfg = params[0], params[1]

    if not os.path.exists(file_path):
        logging.warning(f"File {file_path} does not exist.")
        return

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
        num_samples=cfg.model.num_samples,
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
        if hasattr(m.visual, "vertex_colors") and m.visual.vertex_colors.shape[1] == 4:
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
        scene.remove_node(cam_node)
        scene.remove_node(light_node)
        path = os.path.join(save_dir, f"{filename}_cam_{i_cam}.jpg")
        imageio.imwrite(path, color)

    r.delete()  # release GPU/CPU
    print(f"Saved images: {save_dir}/{filename}")


def task_arm_render(cfg: DictConfig):
    """
    Rendering and saving images of dual-arm-hand grasps (after filtering).
    Does not render pregrasp and squeeze poses.
    """

    exp_path = os.path.join(cfg.paths.experiments_base, cfg.name)
    source_path = os.path.join(exp_path, cfg.task.source_dir)
    all_grasp_file_lst = glob.glob(os.path.join(source_path, "**/*.npy"), recursive=True)

    ########################  Select grasp data files  ########################

    # Selecting with the specified object codes and grasp idx
    object_code_list = list(cfg.task.object_code_list)
    grasp_idx_lst = list(cfg.task.grasp_idx_lst)
    selected_grasp_file_lst = []
    for grasp_file in all_grasp_file_lst:
        path_obj = Path(grasp_file)
        object_code = path_obj.parent.name
        filename = path_obj.stem
        grasp_match = re.search(r"grasp_(\d+)", filename)
        if grasp_match:
            grasp_idx = int(grasp_match.group(1)) if grasp_match else None  # 3
        else:
            raise ValueError("Invalid filename format!")

        if (object_code in object_code_list) and (grasp_idx in grasp_idx_lst):
            selected_grasp_file_lst.append(grasp_file)

    logging.info(f"Select {len(selected_grasp_file_lst)}/{len(all_grasp_file_lst)} grasp file.")

    ########################  Task  ########################

    cfg_lst = [cfg] * len(selected_grasp_file_lst)
    iterable_params = zip(selected_grasp_file_lst, cfg_lst)

    if cfg.task.debug:
        for params in iterable_params:
            save_grasp_images(params)
    else:
        with multiprocessing.Pool(processes=cfg.n_worker) as pool:
            result_iter = pool.imap_unordered(save_grasp_images, iterable_params)
            results = list(result_iter)

    ########################  Check saved files  ########################

    save_dir = os.path.join(exp_path, cfg.task.save_dir)
    img_lst = glob.glob(os.path.join(save_dir, "**/*.jpg"), recursive=True)
    logging.info(f"Save {len(img_lst)} images in {save_dir}.")
    logging.info("Finish grasp image saving.")

    return
