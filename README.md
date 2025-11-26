# BimanGrasp-Generation

## Installation

You can install everything step by step.

1. **Create and activate Conda environment**

   ```bash
   conda create -n bimangrasp python=3.8 -y
   conda activate bimangrasp
   ```

2. **Install PyTorch** (CUDA 12 + PyTorch 2.1.0)

   ```bash
   conda install pytorch==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia 
   ```

3. **Install PyTorch3D**

   ```bash
   pip install https://github.com/facebookresearch/pytorch3d/archive/refs/tags/V0.7.8.tar.gz
   ```

4. **Install other dependencies**

   ```bash
   conda install -c conda-forge transforms3d trimesh kaleido imageio plotly rtree -y
   pip install scipy tensorboard omegaconf hydra-core pyrender mujoco pyvista
   pip install 'pyglet<2'
   pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.0_cu121.html
   ```

5. **Build and install TorchSDF**

   ```bash
   cd thirdparty/TorchSDF
   bash install.sh
   cd ../..
   ```

6. **Install our customized pytorch\_kinematics**

   ```bash
   cd thirdparty
   git clone git@github.com:DexGrasp-TH/pytorch_kinematics.git

   cd pytorch_kinematics
   pip install -e .
   cd ../..
   ```

7. Link the object folder.
   ```bash
   ln -s <target_object_folder> data/object
   ```

## Usage

### Object Assets

We use the object assets from [BODex](https://pku-epic.github.io/BODex/). You can download the pre-processed object assets `DGN_2k_processed.zip` from [Hugging Face](https://huggingface.co/datasets/JiayiChenPKU/BODex) and organize the unzipped folders as below.
```
data/object/DGN_2k
|- processed_data
|  |- core_bottle_1a7ba1f4c892e2da30711cdbdbc73924
|  |_ ...
|- scene_cfg
|  |- core_bottle_1a7ba1f4c892e2da30711cdbdbc73924
|  |_ ...
|- valid_split
|  |- all.json
|  |_ ...
```

### Hand Preprocess

The parameters of hands should be specified in `BimanGrasp-Optimization/cfg/hand/`.

The code has currently been tested only on the provided Shadow Hand settings, but theoretically it can be applied to any hand.

#### Meshes
The TorchSDF cannot correctly process meshes with acute angles between adjacent faces [Issue](https://github.com/wrc042/TorchSDF#note). Please check the meshes of your hand. To preprocess the hand meshes, you can follow these procedures
1. Cut unnecessary parts in Blender. Add bevel to the sharp edges. (Manually)
2. Use `BimanGrasp-Optimization/scripts/smooth_mesh.py` to smooth and simplify the given mesh. 
3. Use `BimanGrasp-Optimization/scripts/check_mesh_sign.py` to compare the calcuated signs by kaolin and torchSDF to final-check the mesh.
4. If there exist wrong results, check which edge is inappropriate, and modify it in Blender. Then, re-smooth and re-check the mesh through above scripts.

#### Contact point candidates on hands

For example, the contact candidate points on Shadow Hands are specified in `BimanGrasp-Optimization/mjcf/shadow/right_hand_contact_points.json`.

You can manually select contact candidates via GUI, using `BimanGrasp-Optimization/scripts/select_contact_candidates_from_GUI.py`.


### Grasp Synthesis

The grasp synthesis pipeline consists of the following procedures:

1. Synthesize bimanual hand grasps. The grasps involve two hands without arms.
   ```
   $ cd BimanGrasp-Optimization
   $ python main.py task=synthesize name=<EXP_NAME>
   ```
1. Compute the pregrasp and squeeze poses for execution.
   ```
   $ cd BimanGrasp-Optimization
   $ python main.py task=compute_three_poses name=<EXP_NAME>
   ```
1. (Optional) Render images of the grasps (before filtering). You need to specify the object codes and grasp indices to visualize in `BimanGrasp-Optimization/cfg/task/render.yaml`.
   ```
   $ cd BimanGrasp-Optimization
   $ python main.py task=render name=<EXP_NAME>
   ```
1. (Optional) Filter the hand grasps by energy (hand-object peneration, self peneration, and joint bound).
   ```
   $ cd BimanGrasp-Optimization
   $ python main.py task=filter name=<EXP_NAME>
   ```
1. Compute corresponding **arm** configurations of the hand grasps and filter the arm-hand grasps by energy.
   ```
   $ cd BimanGrasp-Optimization
   python main.py task=arm_filter name=<EXP_NAME>
   ```
1. (Optional) Render images of the filtered dual-arm-hand grasps. You need to specify the object codes and grasp indices to visualize in `BimanGrasp-Optimization/cfg/task/arm_render.yaml`.
   ```
   $ cd BimanGrasp-Optimization
   $ python main.py task=arm_render name=<EXP_NAME>
   ```
1. (Optional) Visualize certain grasps via web. You need to specify the grasps to visualize in `BimanGrasp-Optimization/cfg/task/vis_single_grasp.yaml`.
   ```
   $ cd BimanGrasp-Optimization
   $ python main.py task=vis_single_grasp name=<EXP_NAME>
   ```

The parameters of each task are specified in `BimanGrasp-Optimization/cfg/task/`.

### Multi-GPUs
You can divide all the objects into groups and run the program on multiple GPUs by
```
$ cd BimanGrasp-Optimization
python main.py task=<TASK> name=<EXP_NAME>
```
The supported tasks include: synthesize, compute_three_poses, filter, arm_filter.

### Simulation-Based Filtering

To furhter filter the synthesized dual-arm-hand grasps via simulation, please refer to [BimanDexGraspBench](https://github.com/DexGrasp-TH/BimanDexGraspBench).


## Acknowledgments

We would like to express our gratitude to the authors of the following repository, from which we referenced code:
* [BimanGrasp-Generation](https://github.com/Tsunami-kun/BimanGrasp-Generation): We rebuild the code base and modify the implementation to improve the synthesis quality and enable it to work with any hand (although not fully tested).
* [DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet/tree/main): The original source of the pipeline for single-hand grasp synthesis.