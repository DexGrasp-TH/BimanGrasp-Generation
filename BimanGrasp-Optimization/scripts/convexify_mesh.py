import trimesh

# 读取你的原始mesh
mesh = trimesh.load("mjcf/shadow2/assets/th_distal_pst.obj")

# 计算凸包
convex_mesh = mesh.convex_hull

mesh.visual.face_colors = [200, 200, 200, 255]  # 原始mesh灰色，不透明
convex_mesh.visual.face_colors = [255, 0, 0, 50]  # 凸包红色，半透明 (alpha=100)

# 可视化比较
scene = trimesh.Scene([mesh, convex_mesh])
scene.show()

# 导出为新的凸近似mesh
convex_mesh.export("mjcf/shadow2/assets/th_distal_pst_cvx.obj")
