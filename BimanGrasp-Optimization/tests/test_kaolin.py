import torch
import trimesh
import kaolin as kal
from kaolin.ops.mesh import index_vertices_by_faces

# -------------------------------
# 1. Create a sphere mesh with Trimesh
# -------------------------------
sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
verts = torch.tensor(sphere.vertices, dtype=torch.float32, device="cuda")
faces = torch.tensor(sphere.faces, dtype=torch.int64, device="cuda")

print(f"Sphere mesh: {verts.shape[0]} vertices, {faces.shape[0]} faces")

# -------------------------------
# 2. Sample query points around the sphere
# -------------------------------
grid_lin = torch.linspace(-1.5, 1.5, 21, device="cuda")
X, Y, Z = torch.meshgrid(grid_lin, grid_lin, grid_lin, indexing="ij")
points = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)  # (N, 3)

print(f"Total query points: {points.shape[0]}")

# -------------------------------
# 3. Compute unsigned distances
# -------------------------------
face_vertices = index_vertices_by_faces(verts[None, ...], faces)
distances, index, dist_type = kal.metrics.trianglemesh.point_to_mesh_distance(points[None, ...], face_vertices)
distances = distances.squeeze(0)  # (N,)

distances = torch.sqrt(distances)

# -------------------------------
# 4. Compute sign (inside/outside)
# -------------------------------
signs = kal.ops.mesh.check_sign(verts[None, ...], faces, points[None, ...])  # (N,) -> bool
sdf = distances * torch.where(signs, -1.0, 1.0)

# -------------------------------
# 5. Print some results
# -------------------------------
for i in range(0, len(points)):
    p = points[i].tolist()
    print(f"Point {p}: SDF = {sdf[0, i].item():.4f}")

print("\n✅ Done: Kaolin SDF computation with Trimesh sphere successful.")

# -------------------------------
# 6. Optional visualization (if matplotlib available)
# -------------------------------
try:
    import matplotlib.pyplot as plt

    # Pick a 2D slice at z ≈ 0
    z_mask = points[:, 2].abs() < 0.05
    slice_points = points[z_mask].cpu().numpy()
    slice_sdf = sdf[0, z_mask].cpu().numpy()

    plt.figure(figsize=(6, 5))
    plt.scatter(slice_points[:, 0], slice_points[:, 1], c=slice_sdf, cmap="coolwarm", s=10)
    plt.colorbar(label="SDF value")
    plt.title("SDF slice at z ≈ 0")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.show()
except ImportError:
    print("matplotlib not installed — skipping visualization.")
