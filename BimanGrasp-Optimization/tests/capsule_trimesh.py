import trimesh as tm
import numpy as np

# -----------------------------
# Create ground plane
# -----------------------------
ground = tm.creation.box(extents=[5, 5, 0.1])  # 5x5 plane, 0.1 thick
ground.apply_translation([0, 0, -0.05])  # move bottom to z=0
ground.visual.face_colors = [200, 200, 200, 255]

# -----------------------------
# Create capsule
# -----------------------------
radius = 0.1
height = 1.0  # full height along z-axis
capsule = tm.creation.capsule(radius=radius, height=height)
capsule.apply_translation([0, 0, 0])  # put base at z=0
capsule.visual.face_colors = [0, 0, 255, 255]

# -----------------------------
# Combine meshes
# -----------------------------
scene = tm.Scene([ground, capsule])

# -----------------------------
# Visualize
# -----------------------------
scene.show()
