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
import pyrender
import multiprocessing

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.common import setup_device, TRANSLATION_NAMES, ROTATION_NAMES
from utils.bimanual_handler import build_bimanual_pose
from utils.visualization import look_at, create_colored_axes


def save_grasp_images(params):
    file_path, cfg, vis_start_pose = params[0], params[1], params[2]

    grasp_idx_lst = cfg.task.grasp_idx_lst
    exp_path = os.path.join(cfg.paths.experiments_base, cfg.name)
    device = setup_device(cfg.gpu)

    filename = os.path.basename(file_path)
    match = re.match(r"(.+)_(\d+)\.npy$", filename)
    if match:
        object_code, opt_step = match.groups()
    else:
        raise ValueError("Invalid filename format!")

    # Create right hand model
    right_hand_model = HandModel(
        handedness="right_hand",
        mjcf_path=cfg.hand.paths.right_hand_mjcf,
        contact_points_path=cfg.hand.paths.right_contact_points,
        device=device,
        n_surface_points=cfg.model.n_surface_points,
        sdf_tool=cfg.model.hand_sdf_tool,
        thumb_links=cfg.hand.hand_params.right_thumb_links,
    )
    left_hand_model = HandModel(
        handedness="left_hand",
        mjcf_path=cfg.hand.paths.left_hand_mjcf,
        contact_points_path=cfg.hand.paths.left_contact_points,
        device=device,
        n_surface_points=cfg.model.n_surface_points,
        sdf_tool=cfg.model.hand_sdf_tool,
        thumb_links=cfg.hand.hand_params.left_thumb_links,
    )
    object_model = ObjectModel(
        data_root_path=cfg.paths.data_root_path,
        batch_size_each=1,
        num_samples=cfg.model.num_samples,
        device=device,
        size="large",
        bodex_format=True,
    )
    object_model.initialize([object_code])

    right_hand_pose_lst = []
    left_hand_pose_lst = []
    opt_step_lst = []
    obj_scale_lst = []
    grasp_id_lst = []

    if not os.path.exists(file_path):
        logging.warning(f"File {file_path} does not exist.")

    ############################### Load data ###############################

    data_dict_lst = np.load(file_path, allow_pickle=True)
    for grasp_idx in grasp_idx_lst:
        data_dict = data_dict_lst[grasp_idx]
        obj_scale = data_dict["scale"]

        # --- Load qpos and construct hand poses ---
        if vis_start_pose and "qpos_right_st" in data_dict:
            right_hand_pose_st, left_hand_pose_st = build_bimanual_pose(
                data_dict["qpos_right_st"],
                data_dict["qpos_left_st"],
                TRANSLATION_NAMES,
                ROTATION_NAMES,
                right_hand_model.get_joint_names(),
                left_hand_model.get_joint_names(),
                device,
            )
            right_hand_pose_lst.append(right_hand_pose_st)
            left_hand_pose_lst.append(left_hand_pose_st)
            opt_step_lst.append(0)
            obj_scale_lst.append(obj_scale)
            grasp_id_lst.append(grasp_idx)

        right_hand_pose, left_hand_pose = build_bimanual_pose(
            data_dict["qpos_right"],
            data_dict["qpos_left"],
            TRANSLATION_NAMES,
            ROTATION_NAMES,
            right_hand_model.get_joint_names(),
            left_hand_model.get_joint_names(),
            device,
        )
        right_hand_pose_lst.append(right_hand_pose)
        left_hand_pose_lst.append(left_hand_pose)
        opt_step_lst.append(opt_step)
        obj_scale_lst.append(obj_scale)
        grasp_id_lst.append(grasp_idx)

    ############################### Rendering ###############################

    right_hand_model.set_parameters(torch.stack(right_hand_pose_lst, dim=0).to(device))
    left_hand_model.set_parameters(torch.stack(left_hand_pose_lst, dim=0).to(device))

    for i_plot in range(len(right_hand_pose_lst)):
        grasp_idx = grasp_id_lst[i_plot]
        obj_scale = obj_scale_lst[i_plot]
        opt_step = opt_step_lst[i_plot]

        right_hand_mesh = right_hand_model.get_trimesh_data(
            i=i_plot, rgba=[0.467, 0.533, 0.600, 0.7], with_contact_points=False, with_axes=False
        )
        # left_hand_mesh = left_hand_model.get_trimesh_data(
        #     i=i_plot, rgba=[0.941, 0.502, 0.502, 0.7], with_contact_points=False, with_axes=False
        # )
        left_hand_mesh = left_hand_model.get_trimesh_data(
            i=i_plot, rgba=[0.467, 0.533, 0.600, 0.7], with_contact_points=False, with_axes=False
        )
        object_model.object_scale_tensor[0] = obj_scale
        # object_mesh = object_model.get_trimesh_data(i=0, rgba=[1.0, 0.961, 0.933, 0.5])
        object_mesh = object_model.get_trimesh_data(i=0, rgba=[1.0, 0.498, 0.314, 0.5])
        axis_mesh = create_colored_axes(origin_size=0.005, axis_length=1.0, radius=0.002)

        # all_meshes = right_hand_mesh + left_hand_mesh + [object_mesh] + axis_mesh
        all_meshes = right_hand_mesh + left_hand_mesh + [object_mesh]

        # Create pyrender scene
        # scene = pyrender.Scene(bg_color=[226 / 255, 240 / 255, 217 / 255, 1.0])
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0])

        # Add all mesh
        for m in all_meshes:
            if hasattr(m.visual, "vertex_colors") and m.visual.vertex_colors.shape[1] == 4:
                color = m.visual.vertex_colors[0] / 255.0
            else:
                color = [0.8, 0.8, 0.8, 1.0]
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=color, metallicFactor=0.0, roughnessFactor=0.9
            )
            mesh_pyr = pyrender.Mesh.from_trimesh(m, material=material)
            scene.add(mesh_pyr)

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)

        os.environ["PYOPENGL_PLATFORM"] = "egl"  # use EGL on headless server
        r = pyrender.OffscreenRenderer(viewport_width=1920, viewport_height=1080)

        save_dir = os.path.join(exp_path, cfg.task.save_dir, f"{object_code}/grasp_{grasp_idx}")
        os.makedirs(save_dir, exist_ok=True)

        cam_pose_lst = [
            [0.7, 0.7, 0.7],
            [0.7, -0.7, 0.7],
            [-0.7, 0.7, 0.7],
            [-0.7, -0.7, -0.7],
        ]
        for i_cam, cam_pose in enumerate(cam_pose_lst):
            cam_pose = look_at(cam_pose, [0, 0, 0])
            cam_node = scene.add(camera, pose=cam_pose)
            light_node = scene.add(light, pose=cam_pose)
            color, _ = r.render(scene)
            scene.remove_node(cam_node)
            scene.remove_node(light_node)
            path = os.path.join(save_dir, f"step_{opt_step}_view_{i_cam}.jpg")
            imageio.imwrite(path, color)

        r.delete()
        logging.info(f"Saved images: {save_dir}/step_{opt_step}, {len(cam_pose_lst)} views")


def task_render(cfg: DictConfig):
    """
    Rendering and saving images of the synthesized bimanual grasps (before filtering).
    Rendering intermediate optimization results.
    No pregrasp and squeeze poses. No arms.
    """

    opt_step_lst = list(cfg.task.opt_step_lst)
    object_code_list = list(cfg.task.object_code_list)

    exp_path = os.path.join(cfg.paths.experiments_base, cfg.name)
    result_path = os.path.join(exp_path, cfg.task.source_dir)
    all_grasp_file_lst = glob.glob(os.path.join(result_path, "**/*.npy"), recursive=True)

    # Select grasp data files with the specified object codes and opt steps in .yaml
    selected_grasp_file_lst = []
    vis_start_pose_lst = []
    for grasp_file in all_grasp_file_lst:
        filename = os.path.basename(grasp_file)
        match = re.match(r"(.+)_(\d+)\.npy$", filename)
        if match:
            object_code, opt_step = match.groups()
            opt_step = int(opt_step)  # str2int
        else:
            raise ValueError("Invalid filename format!")

        if (object_code in object_code_list) and (opt_step in opt_step_lst):
            selected_grasp_file_lst.append(grasp_file)
            # for the file with the smallest opt step, also save the image of the start pose
            vis_start_pose_lst.append(opt_step == opt_step_lst[0])

    logging.info(f"Select {len(selected_grasp_file_lst)}/{len(all_grasp_file_lst)} grasp file.")

    ########################  Task  ########################

    cfg_lst = [cfg] * len(selected_grasp_file_lst)
    iterable_params = zip(selected_grasp_file_lst, cfg_lst, vis_start_pose_lst)

    if cfg.task.debug:
        for params in iterable_params:
            save_grasp_images(params)
    else:
        with multiprocessing.Pool(processes=cfg.n_worker) as pool:
            result_iter = pool.imap_unordered(save_grasp_images, iterable_params)
            results = list(result_iter)

    ########################  Check saved files  ########################

    save_dir = os.path.join(exp_path, "visualizations/opt_process")
    img_lst = glob.glob(os.path.join(save_dir, "**/*.jpg"), recursive=True)
    logging.info(f"Save {len(img_lst)} images in {save_dir}.")
    logging.info("Finish grasp image saving.")

    return
