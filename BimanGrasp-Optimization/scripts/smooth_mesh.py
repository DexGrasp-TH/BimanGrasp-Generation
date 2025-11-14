"""
Preprocess the hand mesh to avoid wrong results from TorchSDF.

Note that TorchSDF cannot accurately process meshes with acute angles between adjacent faces,
owing to its calculation method of the distance sign.

Step:
1. Dense the original mesh.
2. Smooth the mesh to avoid acute anges.
3. Get the convex hull of the mesh.
4. Simplify the mesh to the target number of faces.
"""

import trimesh
import trimesh.smoothing as smoothing
import copy
import open3d as o3d
import numpy as np

# hyper-parameters
meshpath = "mjcf/shadow2/assets/f_distal_pst_cut4.obj"
proc_meshpath = "mjcf/shadow2/assets/f_distal_pst_cut_sm.obj"
target_faces = 500  # final mesh

mesh = trimesh.load(meshpath)

################### Process the mesh #####################

# Get axis-aligned bounding box
bbox = mesh.bounds  # shape (2,3): [[min_x, min_y, min_z], [max_x, max_y, max_z]]
origin = np.mean(bbox, axis=0)
# Compute size along each axis
bbox_size = bbox[1] - bbox[0]
print("Bounding box min:", bbox[0])
print("Bounding box max:", bbox[1])
print("Bounding box size (x, y, z):", bbox_size)

# # convex
# mesh = mesh.convex_hull

# Re-triangulate (improve distribution)
dense_mesh = mesh.subdivide_to_size(max_edge=0.5)

# Smooth
smooth_mesh = copy.copy(dense_mesh)
smoothing.filter_taubin(smooth_mesh, lamb=1.0, iterations=20)
# smoothing.filter_laplacian(smooth_mesh, lamb=1.0, iterations=20)

# Convex
smooth_mesh = smooth_mesh.convex_hull

# Simplify
# Convert to Open3D mesh
o3d_mesh = o3d.geometry.TriangleMesh(
    o3d.utility.Vector3dVector(smooth_mesh.vertices), o3d.utility.Vector3iVector(smooth_mesh.faces)
)
# Simplify with quadratic decimation

simplified_mesh_o3d = o3d_mesh.simplify_quadric_decimation(target_faces)
# Convert back to trimesh
simplified_mesh = trimesh.Trimesh(
    vertices=np.asarray(simplified_mesh_o3d.vertices), faces=np.asarray(simplified_mesh_o3d.triangles)
)

print("Number of original faces:", len(mesh.faces))
print("Number of simplified faces:", len(simplified_mesh.faces))

final_mesh = copy.copy(simplified_mesh)
# Compute dihedral angles (in radians)
dihedral_angles = final_mesh.face_adjacency_angles  # array of angles between adjacent faces' normals
# Check for acute angles
acute_mask = dihedral_angles >= np.deg2rad(85)  # tighter than 90 degree
if np.any(acute_mask):
    print("!!! Mesh has acute angles !!!")
    print("Number of acute angles:", np.sum(acute_mask))
else:
    print("No acute angles in mesh.")

################### Visualization #####################

mesh.visual.face_colors = [200, 200, 200, 255]  # 原始mesh灰色，不透明
dense_mesh.visual.face_colors = [200, 200, 200, 50]  # 凸包红色，半透明 (alpha=100)
smooth_mesh.visual.face_colors = [0, 255, 0, 100]  # 凸包红色，半透明 (alpha=100)
simplified_mesh.visual.face_colors = [0, 0, 255, 100]  # 凸包红色，半透明 (alpha=100)

t = bbox_size[0] + 5
dense_mesh.apply_translation([t, 0, 0])
smooth_mesh.apply_translation([2 * t, 0, 0])
simplified_mesh.apply_translation([3 * t, 0, 0])

vertex_cloud = trimesh.points.PointCloud(mesh.vertices, colors=[255, 0, 0, 255])
dense_vertex_cloud = trimesh.points.PointCloud(dense_mesh.vertices, colors=[255, 0, 0, 255])
smooth_vertex_cloud = trimesh.points.PointCloud(smooth_mesh.vertices, colors=[255, 0, 0, 255])
simplified_vertex_cloud = trimesh.points.PointCloud(simplified_mesh.vertices, colors=[255, 0, 0, 255])

scene = trimesh.Scene(
    [
        mesh,
        vertex_cloud,
        dense_mesh,
        dense_vertex_cloud,
        smooth_mesh,
        smooth_vertex_cloud,
        simplified_mesh,
        simplified_vertex_cloud,
    ]
)
scene.show()

final_mesh.export(proc_meshpath)
print(f"Save processed mesh to {proc_meshpath}.")
