import open3d as o3d, numpy as np, matplotlib.pyplot as plt

dataset = o3d.data.EaglePointCloud()
pcd = o3d.io.read_point_cloud(dataset.path)

r = o3d.visualization.rendering.OffscreenRenderer(640, 480)
mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = "defaultUnlit"
mat.point_size = 2
r.scene.add_geometry("pcd", pcd, mat)
r.scene.camera.look_at([0, 0, 0], [0, 0, 2], [0, 1, 0])
img = np.asarray(r.render_to_image())

plt.imshow(img)
plt.axis("off")
plt.show()
