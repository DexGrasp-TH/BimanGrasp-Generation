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

   pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.0_cu121.html

   # use blender to smooth hand mesh
   pip install bpy
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

```bash
# Generate bimanual grasps
python main.py

# Visualize results
python visualization.py --object_code <object_name> --num <grasp_index>
```

While BimanGrasp-Generation is able to work with any 3D object mesh, this repository contains a mini demo on 5 randomly sample objects. To prepare your own objects, you could follow the asset processing script and instructions by DexGraspNet: https://github.com/PKU-EPIC/DexGraspNet/tree/main/asset_process

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
