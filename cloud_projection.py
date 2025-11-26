import os
import glob
import numpy as np
import pandas as pd
import joblib
import open3d as o3d
import cv2
import subprocess
import tempfile
import json

# ============================================================
# CONFIG
# ============================================================

PLY_DIR = "data/point_cloud/test"
MODEL_PATH = "models/qoe_model_xgb_v2.0.pkl"
OUTPUT_CSV = "data/pc_projection_qoe.csv"

# Subsampling target point budgets
POINT_BUDGETS = [5000, 10000, 20000, 40000, 80000]

# Projection settings
IMG_W, IMG_H = 512, 512
POINT_SIZE = 3

# Fixed camera views: (center, eye, up)
def get_camera_pose(center, D, view):
    if view == "front":
        return center + np.array([0, +D, 0]), center, np.array([0, 0, 1])
    if view == "back":
        return center + np.array([0, -D, 0]), center, np.array([0, 0, 1])
    if view == "right":
        return center + np.array([+D, 0, 0]), center, np.array([0, 0, 1])
    if view == "left":
        return center + np.array([-D, 0, 0]), center, np.array([0, 0, 1])
    if view == "top":
        return center + np.array([0, 0, -D]), center, np.array([0, -1, 0])
    if view == "bottom":
        return center + np.array([0, 0, +D]), center, np.array([0, 1, 0])

VIEWS = ["front", "back", "right", "left", "top", "bottom"]

# ============================================================
# LOAD MODEL
# ============================================================

print(f"Loading QoE model from {MODEL_PATH} ...")
model = joblib.load(MODEL_PATH)

# The feature order must match your training script
FEATURE_COLS = [
    "vmaf",
    "psnr",
    "ssim",
    "strred",
    "bitrate",
    "is_rebuffered",
    "spatial_info",
    "temporal_info",
]

# ============================================================
# RENDERING HELPERS
# ============================================================

def make_renderer():
    return o3d.visualization.rendering.OffscreenRenderer(IMG_W, IMG_H)

def render_view(renderer, pcd, eye, center, up):
    renderer.scene.clear_geometry()
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = POINT_SIZE
    renderer.scene.add_geometry("pcd", pcd, mat)
    renderer.scene.camera.look_at(center, eye, up)
    img = np.asarray(renderer.render_to_image())[:, :, :3]
    return img

def render_all_views(pcd):
    imgs = []

    # Compute geometry-based camera distance
    center = pcd.get_center()
    extent = pcd.get_axis_aligned_bounding_box().get_extent()
    D = float(max(extent)) * 1.2   # add padding so the object stays inside view

    renderer = o3d.visualization.rendering.OffscreenRenderer(IMG_W, IMG_H)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = POINT_SIZE

    for view in VIEWS:
        cam_pos, lookat, up = get_camera_pose(center, D, view)

        renderer.scene.clear_geometry()
        renderer.scene.add_geometry("pcd", pcd, mat)

        renderer.scene.camera.look_at(
            lookat.tolist(),
            cam_pos.tolist(),
            up.tolist()
        )

        img = np.asarray(renderer.render_to_image())[:, :, :3]
        imgs.append(img)

    return imgs

# ============================================================
# METRICS: SSIM, PSNR, VMAF, SI, TI
# ============================================================

def rgb_to_y(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)[:, :, 0]

def ssim_psnr_multi(ref_imgs, dist_imgs):
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr

    ssim_vals, psnr_vals = [], []
    for rimg, dimg in zip(ref_imgs, dist_imgs):
        y_ref, y_dist = rgb_to_y(rimg), rgb_to_y(dimg)
        ssim_vals.append(ssim(y_ref, y_dist, data_range=255))
        psnr_vals.append(psnr(y_ref, y_dist, data_range=255))
    return float(np.mean(ssim_vals)), float(np.mean(psnr_vals))

def vmaf_one_view(ref_img, dist_img):
    """Compute VMAF for a single image pair via ffmpeg/libvmaf. Returns NaN if fails."""
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ref_path = os.path.join(tmp, "ref.png")
            dist_path = os.path.join(tmp, "dist.png")
            json_path = os.path.join(tmp, "vmaf.json")

            cv2.imwrite(ref_path, cv2.cvtColor(ref_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(dist_path, cv2.cvtColor(dist_img, cv2.COLOR_RGB2BGR))

            cmd = [
                "ffmpeg", "-y",
                "-i", ref_path,
                "-i", dist_path,
                "-lavfi",
                f"[0:v]scale={IMG_W}:{IMG_H},format=yuv420p[ref];"
                f"[1:v]scale={IMG_W}:{IMG_H},format=yuv420p[dist];"
                f"[ref][dist]libvmaf=log_path={json_path}:log_fmt=json",
                "-f", "null", "-"
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

            with open(json_path, "r") as f:
                data = json.load(f)
            return float(data["frames"][0]["metrics"]["vmaf"])
    except Exception as e:
        print("VMAF computation failed, returning NaN:", e)
        return np.nan

def vmaf_multi(ref_imgs, dist_imgs):
    vals = []
    for rimg, dimg in zip(ref_imgs, dist_imgs):
        vals.append(vmaf_one_view(rimg, dimg))
    return float(np.nanmean(vals))

def spatial_temporal_info(ref_imgs_seq):
    """
    Compute simple per-frame SI/TI based on the first view's luma.
    SI: std of Sobel magnitude
    TI: std of frame-to-frame difference in luma
    """
    si_list = []
    ti_list = []
    prev_y = None

    for img in ref_imgs_seq:
        y = rgb_to_y(img)
        # spatial information
        gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx*gx + gy*gy)
        si = float(np.std(mag))
        si_list.append(si)

        # temporal information
        if prev_y is None:
            ti = 0.0
        else:
            diff = y.astype(np.float32) - prev_y.astype(np.float32)
            ti = float(np.std(diff))
        ti_list.append(ti)
        prev_y = y

    return si_list, ti_list

# ============================================================
# SUBSAMPLING
# ============================================================

def subsample_points(pcd, target_points):
    pts = np.asarray(pcd.points)
    n = len(pts)
    if target_points >= n:
        return pcd, n

    ratio = target_points / n
    pcd_sub = pcd.random_down_sample(ratio)
    return pcd_sub, len(pcd_sub.points)

# ============================================================
# MAIN
# ============================================================

def main():
    frame_paths = sorted(glob.glob(os.path.join(PLY_DIR, "*.ply")))
    if not frame_paths:
        print(f"No .ply files found in {PLY_DIR}")
        return

    print(f"Found {len(frame_paths)} point-cloud frames.")

    if not True:  # turn off when done testing
        path = frame_paths[0]
        pcd = o3d.io.read_point_cloud(path)
        # imgs = render_all_views(pcd)

        import matplotlib.pyplot as plt

        for target_pts in [5000, 20000, 80000]:
            pcd_sub, _ = subsample_points(pcd, target_pts)
            imgs = render_all_views(pcd_sub)

            plt.figure(figsize=(10, 6))
            for i, img in enumerate(imgs):
                plt.subplot(2, 3, i+1)
                plt.imshow(img)
                plt.title(f"{VIEWS[i]} ({target_pts} pts)")
                plt.axis("off")
            plt.tight_layout()
            plt.show()
        
        return  # stop here

    # First pass: get max point count to define bitrate proxy scale
    max_points = 0
    frame_point_counts = []
    for path in frame_paths:
        pcd = o3d.io.read_point_cloud(path)
        n_points = len(pcd.points)
        frame_point_counts.append(n_points)
        max_points = max(max_points, n_points)

    if max_points == 0:
        print("All frames have zero points; aborting.")
        return

    print(f"Max points in any frame: {max_points}")

    # Second pass: projection + QoE prediction
    results = []

    # Precompute reference projections for SI/TI
    print("Rendering reference projections for SI/TI ...")
    ref_view_seq = []
    for path in frame_paths:
        pcd = o3d.io.read_point_cloud(path)
        ref_views = render_all_views(pcd)
        ref_view_seq.append(ref_views[0])  # use first view for SI/TI

    si_list, ti_list = spatial_temporal_info(ref_view_seq)

    print("Processing frames for QoE predictions ...")
    for idx, path in enumerate(frame_paths):
        frame_name = os.path.basename(path)
        print(f"\nFrame {idx+1}/{len(frame_paths)}: {frame_name}")

        full_pcd = o3d.io.read_point_cloud(path)
        full_views = render_all_views(full_pcd)

        # Use SI/TI from first-view sequence
        spatial_info = si_list[idx]
        temporal_info = ti_list[idx]

        # Iterate over subsampling budgets
        for target_pts in POINT_BUDGETS:
            print(f"  Subsampling to ~{target_pts} points ...")
            pcd_sub, actual_pts = subsample_points(full_pcd, target_pts)
            dist_views = render_all_views(pcd_sub)

            # Compute 2D metrics
            ssim_val, psnr_val = ssim_psnr_multi(full_views, dist_views)
            # vmaf_val = vmaf_multi(full_views, dist_views)

            vmaf_val = 40.0
            # psnr_val = 32.0
            # ssim_val = 0.9

            # Bitrate proxy: scale points to ~[0, 3000] kbps range
            bitrate_proxy = (actual_pts / max_points) * 3000.0

            # No stalls in this offline experiment
            is_rebuffered = 0.0

            # STRRED is not computed for projections; set NaN
            strred_val = np.nan

            # Build features in the exact training order
            feat_vec = np.array([[
                vmaf_val,
                psnr_val,
                ssim_val,
                strred_val,
                bitrate_proxy,
                is_rebuffered,
                spatial_info,
                temporal_info,
            ]], dtype=float)

            qoe_pred = float(model.predict(feat_vec)[0])

            results.append({
                "frame_idx": idx,
                "frame_name": frame_name,
                "target_points": target_pts,
                "actual_points": actual_pts,
                "vmaf_proj": vmaf_val,
                "psnr_proj": psnr_val,
                "ssim_proj": ssim_val,
                "bitrate_proxy": bitrate_proxy,
                "spatial_info_proj": spatial_info,
                "temporal_info_proj": temporal_info,
                "qoe_pred": qoe_pred,
            })

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved projection-based QoE results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
