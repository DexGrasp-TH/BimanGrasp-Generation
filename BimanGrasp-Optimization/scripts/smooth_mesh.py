# import bpy

meshpath= "mjcf/shadow2/assets/th_distal_pst.obj"

# bpy.ops.import_scene.obj(filepath=meshpath)

# obj = bpy.context.selected_objects[0]

# # 1. Smooth shading
# bpy.ops.object.shade_smooth()

# # 2. Remove sharp edges (crease angle < 30°)
# bpy.ops.mesh.select_all(action='SELECT')
# bpy.ops.mesh.edges_select_sharp(sharpness=0.523599)  # 30 degrees
# bpy.ops.mesh.bevel(offset=0.002)   # small bevel

# # 3. Remesh (voxel remesh)
# bpy.ops.object.modifier_add(type='REMESH')
# obj.modifiers["Remesh"].mode = 'SMOOTH'
# obj.modifiers["Remesh"].voxel_size = 0.01
# bpy.ops.object.modifier_apply(modifier='Remesh')

# bpy.ops.export_scene.obj(filepath="mjcf/shadow2/assets/th_distal_pst_sm.obj")

import trimesh
import trimesh.smoothing as smoothing

mesh = trimesh.load(meshpath)

# Laplacian
smoothing.filter_laplacian(mesh, lamb=0.5, iterations=50)

mesh.export("mjcf/shadow2/assets/th_distal_pst_sm.obj")
