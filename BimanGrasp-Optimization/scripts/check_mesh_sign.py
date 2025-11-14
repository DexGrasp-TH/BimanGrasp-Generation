import trimesh
import kaolin
from torchsdf import index_vertices_by_faces, compute_sdf
import os
import torch
from time import time
import numpy as np

device = "cuda:0"

meshpath = "mjcf/shadow2/assets/th_distal_pst_cut_sm.obj"
mesh = trimesh.load(meshpath)

bbox = mesh.bounds  # shape (2,3): [[min_x, min_y, min_z], [max_x, max_y, max_z]]
origin = torch.tensor(np.mean(bbox, axis=0)).to(device).detach().float()
bbox_size = torch.tensor(bbox[1] - bbox[0]).to(device).detach().float()

# Ns
num_sample = 10000000
samples = torch.rand((num_sample, 3)).to(device).detach()
samples = origin.unsqueeze(0) + 2 * (samples * bbox_size.unsqueeze(0) - bbox_size.unsqueeze(0) / 2)

all_pass = True

print("====Sign check====")
print(f"Mesh: {meshpath}")
model_path = meshpath
mesh = trimesh.load(model_path, force="mesh", process=False)
# (Ns, 3)
x = samples.clone().requires_grad_()
# (Nv, 3)
verts = torch.Tensor(mesh.vertices.copy()).to(device)
# (Nf, 3)
faces = torch.Tensor(mesh.faces.copy()).long().to(device)
# (1, Nf, 3, 3)
face_verts = kaolin.ops.mesh.index_vertices_by_faces(verts.unsqueeze(0), faces)
# (Nf, 3, 3)
face_verts_ts = index_vertices_by_faces(verts, faces)

# Kaolin
# (1, Ns)
signs = kaolin.ops.mesh.check_sign(verts.unsqueeze(0), faces, x.unsqueeze(0))
signs = torch.where(signs, -1 * torch.ones_like(signs, dtype=torch.int32), torch.ones_like(signs, dtype=torch.int32))

# TorchSDF
# (Ns)
distances_ts, signs_ts, normals_ts, clst_points_ts = compute_sdf(x, face_verts_ts)
# (1, Ns)
dif = signs_ts != signs
dif = dif.reshape(-1)
miss_points = x[dif, :]
color = torch.zeros_like(miss_points).int()
color[:, 0] = 255

miss_cloud = None
GREEN = "\033[92m"
RED = "\033[91m"
END = "\033[0m"
if miss_points.shape[0] == 0:
    print(f"{GREEN}✓ Passed{END}")
else:
    print(f"{RED}✗ Failed{END}")
    print("Number of miss_points: ", len(miss_points))
    miss_cloud = trimesh.points.PointCloud(miss_points.cpu().detach().numpy())

# build scene
scene = trimesh.Scene()
mesh.visual.face_colors = [200, 200, 200, 100]
scene.add_geometry(mesh)
# 2. add vertices
points = trimesh.points.PointCloud(mesh.vertices, colors=[0, 0, 255, 255])
scene.add_geometry(points)
# 3. add edges
edges = mesh.edges_unique
lines = trimesh.load_path(mesh.vertices[edges])
scene.add_geometry(lines)

if miss_cloud is not None:
    scene.add_geometry(miss_cloud)

# draw part of the query points
N_show = 100
idx = torch.randperm(samples.shape[0])[:N_show]
pts = samples[idx].detach().cpu().numpy()
colors = np.tile(np.array([0, 200, 200, 100]), (pts.shape[0], 1))
query_cloud = trimesh.points.PointCloud(pts, colors=colors)
scene.add_geometry(query_cloud)

scene.show()
