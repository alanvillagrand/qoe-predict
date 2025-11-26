import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("data/point_cloud/ricardo9/ply/frame0000.ply")
center = pcd.get_center()
extent = pcd.get_axis_aligned_bounding_box().get_extent()
D = max(extent)  # distance offset

def get_camera_pose(view):
    if view == "front":
        return center + np.array([0, +D, 0]), center, np.array([0, 0, 1])
    if view == "back":
        return center + np.array([0, -D, 0]), center, np.array([0, 0, 1])
    if view == "right":
        return center + np.array([+D, 0, 0]), center, np.array([0, 0, 1])
    if view == "left":
        return center + np.array([-D, 0, 0]), center, np.array([0, 0, 1])
    if view == "bottom":
        return center + np.array([0, 0, +D]), center, np.array([0, 1, 0])
    if view == "top":
        return center + np.array([0, 0, -D]), center, np.array([0, -1, 0])

def set_camera(vis, view="front"):
    ctr = vis.get_view_control()

    camera_position, lookat, up = get_camera_pose(view)
    front = lookat - camera_position
    front /= np.linalg.norm(front)

    ctr.set_front(front.tolist())
    ctr.set_lookat(lookat.tolist())
    ctr.set_up(up.tolist())
    ctr.set_zoom(0.8)
    return False

# Example: render top view
o3d.visualization.draw_geometries_with_animation_callback(
    [pcd],
    lambda vis: set_camera(vis, view="top")
)