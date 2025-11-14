import trimesh
import copy
import numpy as np

# hyper-parameters
meshpath = "/data/dataset/BODex/object/DGN_2k/processed_data/ddg_gd_box_poisson_005/mesh/simplified.obj"
target_faces = 500  # final mesh

mesh = trimesh.load(meshpath)

final_mesh = copy.copy(mesh)
# Compute dihedral angles (in radians)
dihedral_angles = final_mesh.face_adjacency_angles  # array of angles between adjacent faces' normals
# Check for acute angles
acute_mask = dihedral_angles >= (np.pi / 2) - 1e-2
if np.any(acute_mask):
    print("!!! Mesh has acute angles !!!")
    print("Number of acute angles:", np.sum(acute_mask))
else:
    print("No acute angles in mesh.")

################### Visualization #####################

mesh.visual.face_colors = [200, 200, 200, 255]  # 原始mesh灰色，不透明
vertex_cloud = trimesh.points.PointCloud(mesh.vertices, colors=[255, 0, 0, 255])

scene = trimesh.Scene(
    [
        mesh,
        vertex_cloud,
    ]
)
scene.show()
