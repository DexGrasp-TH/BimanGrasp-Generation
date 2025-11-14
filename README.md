# BimanGrasp-Generation
---

## Installation

You can install everything step by step.

1. **Create and activate Conda environment**

   ```bash
   conda create -n bimangrasp python=3.8 -y
   conda activate bimangrasp
   ```

2. **Install PyTorch (CUDA 12 support)**

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
   pip install urdf_parser_py scipy networkx tensorboard six omegaconf hydra-core 
   pip install mujoco
   pip install 'pyglet<2'

   # for developing
   pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.0_cu121.html

   ```

5. **Build and install TorchSDF**

   ```bash
   cd thirdparty/TorchSDF
   bash install.sh
   cd ../..
   ```

6. **Install pytorch\_kinematics**

   ```bash
   cd thirdparty
   git clone git@github.com:DexGrasp-TH/pytorch_kinematics.git

   cd pytorch_kinematics
   pip install -e .
   cd ../..
   ```

---

7. Link the object folder.
   ```bash
   ln -s <target_object_folder> data/object
   ```

## Usage

### Hand Mesh Preprocess
The TorchSDF cannot correctly process meshes with acute angles between adjacent faces [Issue](https://github.com/wrc042/TorchSDF#note). Please check the meshes of your hand. To preprocess the hand meshes, you can follow these procedures
1. Cut unnecessary parts in Blender. Add bevel to the sharp edges. (Manually)
2. Use `scripts/smooth_mesh.py` to smooth and simplify the given mesh. 
3. Use `scripts/check_mesh_sign.py` to compare the calcuated signs by kaolin and torchSDF to final-check the mesh.
4. If there exist wrong results, check which edge is inappropriate, and modify it in Blender. Then, re-smooth and re-check the mesh through above scripts.

### Grasp Synthesis


## Demo Visualization

This demo visualizations on the 5 objects are tested on an A40 GPU without cherry picking.


| | | | |
|:---:|:---:|:---:|:---:|
| <img src="assets\figs\Breyer_Horse_Of_The_Year_2015_0_screenshot.png" width="100%"> | <img src="assets\figs\Breyer_Horse_Of_The_Year_2015_1_screenshot.png" width="100%"> | <img src="assets\figs\Breyer_Horse_Of_The_Year_2015_2_screenshot.png" width="100%"> | <img src="assets\figs\Breyer_Horse_Of_The_Year_2015_3_screenshot.png" width="100%"> |
| <img src="assets\figs\Cole_Hardware_Dishtowel_Multicolors_0_screenshot.png" width="100%"> | <img src="assets\figs\Cole_Hardware_Dishtowel_Multicolors_1_screenshot.png" width="100%"> | <img src="assets\figs\Cole_Hardware_Dishtowel_Multicolors_2_screenshot.png" width="100%"> | <img src="assets\figs\Cole_Hardware_Dishtowel_Multicolors_3_screenshot.png" width="100%"> |
| <img src="assets\figs\Curver_Storage_Bin_Black_Small_0_screenshot.png" width="100%"> | <img src="assets\figs\Curver_Storage_Bin_Black_Small_1_screenshot.png" width="100%"> | <img src="assets\figs\Curver_Storage_Bin_Black_Small_2_screenshot.png" width="100%"> | <img src="assets\figs\Curver_Storage_Bin_Black_Small_3_screenshot.png" width="100%"> |
| <img src="assets\figs\Hasbro_Monopoly_Hotels_Game_0_screenshot.png" width="100%"> | <img src="assets\figs\Hasbro_Monopoly_Hotels_Game_1_screenshot.png" width="100%"> | <img src="assets\figs\Hasbro_Monopoly_Hotels_Game_2_screenshot.png" width="100%"> | <img src="assets\figs\Hasbro_Monopoly_Hotels_Game_3_screenshot.png" width="100%"> |
| <img src="assets\figs\Schleich_S_Bayala_Unicorn_70432_0_screenshot.png" width="100%"> | <img src="assets\figs\Schleich_S_Bayala_Unicorn_70432_1_screenshot.png" width="100%"> | <img src="assets\figs\Schleich_S_Bayala_Unicorn_70432_2_screenshot.png" width="100%"> | <img src="assets\figs\Schleich_S_Bayala_Unicorn_70432_3_screenshot.png" width="100%"> |


## To-Do

- [ ] Release validation code for both simulation and real-world validation.

## Acknowledgments

We would like to express our gratitude to the authors of the following repository, from which we referenced code:

* [DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet/tree/main)

## Dataset Repo

Our released BimanGrasp-Dataset is in this repo: [[BimanGrasp-Dataset](https://github.com/Tsunami-kun/BimanGrasp-Dataset)].

## License
The Project is under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (LICENSE.md).

## Citation

If you find this code useful, please consider citing:

```bibtex
@article{shao2024bimanual,
  title={Bimanual grasp synthesis for dexterous robot hands},
  author={Shao, Yanming and Xiao, Chenxi},
  journal={IEEE Robotics and Automation Letters},
  year={2024},
  publisher={IEEE}
}
```
