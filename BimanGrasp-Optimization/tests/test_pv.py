# import pyvista as pv

# capsule = pv.Cylinder(radius=0.05, height=0.2, direction=(0, 0, 1)).triangulate()
# picked_points = []


# def callback(picker, event):
#     point = picker.pick_position
#     picked_points.append(point)
#     print(f"Picked on surface: {point}")


# plotter = pv.Plotter()
# plotter.add_mesh(capsule, color="lightblue", opacity=0.6)
# plotter.enable_surface_picking(callback=callback, show_message=True)
# plotter.show()

# print("Picked points:", picked_points)

import pyvista as pv

# Create a coarse capsule-like mesh (e.g., a cylinder with hemispheres)
capsule = pv.Cylinder(radius=0.05, height=0.2, direction=(0, 0, 1)).triangulate()

# Subdivide the mesh to get denser vertices
capsule_dense = capsule.subdivide(3)  # increase number for denser surface

# Interactive picking
picked_points = []


def callback(point, event):
    picked_points.append(point)
    print(f"Picked: {point}")


plotter = pv.Plotter()
plotter.add_mesh(capsule_dense, color="lightblue", opacity=0.6, show_edges=True)
plotter.enable_point_picking(callback=callback, use_mesh=True, show_message=True)
plotter.show()

print("Picked points:", picked_points)
