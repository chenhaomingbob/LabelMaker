# wheeltec2labelmaker.py

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from bisect import bisect_left
from tqdm import tqdm
import open3d as o3d
from PIL import Image
# from ..utils_3d import fuse_mesh
from utils_3d import fuse_mesh

debug = False


# =================================================================================
# Helper functions and constants from testwheeltec.py
# =================================================================================

def quaternion_to_rotation_matrix(q):
    """Converts a quaternion to a rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x ** 2 - 2 * y ** 2]
    ])


def create_transform_matrix(t, q):
    """Creates a 4x4 transformation matrix from translation and quaternion."""
    transform = np.eye(4)
    transform[:3, :3] = quaternion_to_rotation_matrix(q)
    transform[:3, 3] = t
    return transform


def pose_to_homogeneous_matrix(tran_x, tran_y, tran_z, rot_x, rot_y, rot_z, rot_w):
    """Converts pose components to a 4x4 homogeneous transformation matrix."""
    position = np.array([tran_x, tran_y, tran_z])
    rotation_matrix = quaternion_to_rotation_matrix([rot_x, rot_y, rot_z, rot_w])
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotation_matrix
    homogeneous_matrix[:3, 3] = position
    return homogeneous_matrix


# --- Static Transforms and Intrinsics (from testwheeltec.py) ---
# NOTE: These are camera-specific and may need adjustment for different datasets.
color_camera_intrinsic_info = np.array([
    [450.8250732421875, 0.0, 334.319580078125],
    [0.0, 450.8250732421875, 247.18316650390625],
    [0.0, 0.0, 1.0]
])

depth_camera_intrinsic_info = np.array(
    [
        [475.4315490722656, 0.0, 324.42584228515625],
        [0.0, 475.4315490722656, 197.35276794433594],
        [0.0, 0.0, 1.0]
    ]
)
# This matrix transforms points from the DEPTH camera frame to the COLOR camera frame.
T_depth_2_camera_color_frame = create_transform_matrix(
    t=np.array([-0.000447075, 0.010095531, 0.0000743030980229377]),
    q=np.array([-0.000136242, 0.001019766, 0.004025249, 0.999991357])
)
# Transforms to get from color camera frame to map (world) frame
# Chain: Camera -> base_footprint -> odom -> map
T_base_footprint_2_camera_link = create_transform_matrix(
    t=np.array([0.32039, 0.00164, 0.13208]),
    q=np.array([0.0, 0.0, 0.0, 1])
)
T_camera_link_2_camera_depth_frame = create_transform_matrix(
    t=np.array([0.0, 0.0, 0.0]),
    q=np.array([-0.5, 0.5, -0.5, 0.5])
)
T_depth_2_camera_color_frame = create_transform_matrix(
    t=np.array([-0.000447075, 0.010095531, 0.0000743030980229377]),
    q=np.array([-0.000136242, 0.001019766, 0.004025249, 0.999991357])
)

# This is the static transform from the color camera optical frame to the robot's base_footprint frame
T_base_footprint_2_camera_color = T_base_footprint_2_camera_link @ T_camera_link_2_camera_depth_frame @ T_depth_2_camera_color_frame

T_laser_2_camera_color = np.array(
    [
        [0.999964, -0.00283691, -0.00805015, -0.00106572],
        [-0.00805091, -0.000257861, -0.999968, -0.383872],
        [0.00283474, 0.999996, -0.000280691, - 0.23023],
        [0, 0, 0, 1]
    ]
)


# =================================================================================
# Data Synchronization (adapted from testwheeltec.py and arkitscenes2labelmaker.py)
# =================================================================================
def find_closest_timestamp(target, timestamps):
    """Finds the index of the closest timestamp in a sorted list."""
    pos = bisect_left(timestamps, target)
    if pos == 0:
        return 0
    if pos == len(timestamps):
        return len(timestamps) - 1
    before = timestamps[pos - 1]
    after = timestamps[pos]
    if after - target < target - before:
        return pos
    else:
        return pos - 1


def match_all_sensors_by_image(image_dir, depth_dir, lidar_dir, odom_file, tf_map_to_odom_file, max_time_diff_sec=0.05):
    """
    Matches sensor data based on image timestamps.
    """
    print("Synchronizing sensor data based on image timestamps...")
    # 1. Get all file lists and read CSVs
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.npy')])
    lidar_files = sorted([f for f in os.listdir(lidar_dir) if f.endswith('.ply')])
    odom_df = pd.read_csv(odom_file)
    tf_df = pd.read_csv(tf_map_to_odom_file)

    # 2. Extract and sort timestamps
    image_timestamps = sorted([float(Path(f).stem) for f in image_files])
    depth_timestamps = sorted([float(Path(f).stem) for f in depth_files])
    lidar_timestamps = sorted([float(Path(f).stem) for f in lidar_files])
    odom_timestamps = sorted(odom_df['timestamp'].unique())
    tf_timestamps = sorted(tf_df['timestamp'].unique())

    if not image_timestamps:
        raise ValueError("No image files found in the specified directory.")

    # 3. Match data for each image timestamp
    matched_data = []
    for image_ts in tqdm(image_timestamps, desc="Matching frames"):
        # Find closest depth map
        depth_idx = find_closest_timestamp(image_ts, depth_timestamps)
        depth_ts = depth_timestamps[depth_idx]

        # Find closest LiDAR scan
        lidar_idx = find_closest_timestamp(image_ts, lidar_timestamps)
        lidar_ts = lidar_timestamps[lidar_idx]

        # Find closest odometry
        odom_idx = find_closest_timestamp(image_ts, odom_timestamps)
        odom_ts = odom_timestamps[odom_idx]

        # Find closest TF
        tf_idx = find_closest_timestamp(image_ts, tf_timestamps)
        tf_ts = tf_timestamps[tf_idx]

        # Check if timestamps are within the allowed difference
        if (abs(image_ts - depth_ts) > max_time_diff_sec or
                abs(image_ts - lidar_ts) > max_time_diff_sec or
                abs(image_ts - odom_ts) > max_time_diff_sec or
                abs(image_ts - tf_ts) > max_time_diff_sec):
            continue

        # Get the corresponding data rows
        odom_row = odom_df[odom_df['timestamp'] == odom_ts].iloc[0]
        tf_row = tf_df[tf_df['timestamp'] == tf_ts].iloc[0]

        matched_data.append({
            'timestamp': image_ts,
            'image_file': os.path.join(image_dir, f"{image_ts}.png"),
            'depth_file': os.path.join(depth_dir, f"{depth_ts}.npy"),
            'lidar_file': os.path.join(lidar_dir, f"{lidar_ts}.ply"),
            'odom_data': odom_row.to_dict(),
            'tf_map2odom_data': tf_row.to_dict(),
        })

    if not matched_data:
        raise ValueError("No matching frames found. Check timestamps and `max_time_diff_sec`.")

    print(f"Found {len(matched_data)} synchronized frames.")
    return pd.DataFrame(matched_data)


def reproject_depth_to_color_view(depth_map,
                                  depth_intr,
                                  color_intr,
                                  depth_to_color_extr,
                                  color_img_shape,
                                  depth_scale=1000.0):
    """
    Reprojects a depth map from the depth camera's view to the color camera's view.

    Args:
        depth_map (np.ndarray): The raw depth map from the depth sensor.
        depth_intr (np.ndarray): 3x3 intrinsics of the depth camera.
        color_intr (np.ndarray): 3x3 intrinsics of the color camera.
        depth_to_color_extr (np.ndarray): 4x4 extrinsic matrix transforming points
                                          from the depth camera frame to the color camera frame.
        color_img_shape (tuple): The (height, width) of the color image.
        depth_scale (float): The factor to convert depth map values to meters (e.g., 1000.0 for mm).

    Returns:
        np.ndarray: A new depth map aligned with the color camera's view, with values in mm (uint16).
    """
    h, w = color_img_shape[:2]

    # 1. Create 3D point cloud in the depth camera's coordinate system
    fx_d, fy_d = depth_intr[0, 0], depth_intr[1, 1]
    cx_d, cy_d = depth_intr[0, 2], depth_intr[1, 2]

    v_d, u_d = np.indices(depth_map.shape)
    z_d = depth_map / depth_scale  # Convert to meters
    x_d = (u_d - cx_d) * z_d / fx_d
    y_d = (v_d - cy_d) * z_d / fy_d

    # Keep only valid points
    valid_mask = z_d > 0
    points_in_depth_cam = np.stack([x_d[valid_mask], y_d[valid_mask], z_d[valid_mask]], axis=-1)

    # 2. Transform points to the color camera's coordinate system
    points_hom = np.hstack([points_in_depth_cam, np.ones((len(points_in_depth_cam), 1))])
    points_in_color_cam = (depth_to_color_extr @ points_hom.T).T[:, :3]

    # 3. Project the 3D points onto the color image plane
    fx_c, fy_c = color_intr[0, 0], color_intr[1, 1]
    cx_c, cy_c = color_intr[0, 2], color_intr[1, 2]

    z_c = points_in_color_cam[:, 2]
    u_c = (points_in_color_cam[:, 0] * fx_c / z_c) + cx_c
    v_c = (points_in_color_cam[:, 1] * fy_c / z_c) + cy_c

    # 4. Filter out points that are behind the camera or outside the image bounds
    valid_proj_mask = (z_c > 0) & (u_c >= 0) & (u_c < w) & (v_c >= 0) & (v_c < h)
    u_valid = u_c[valid_proj_mask].astype(int)
    v_valid = v_c[valid_proj_mask].astype(int)
    z_valid = z_c[valid_proj_mask]

    # 5. Create the new depth image, handling occlusions
    new_depth_image = np.zeros((h, w), dtype=np.float32)

    # Sort points by depth in descending order. This ensures that when multiple
    # points project to the same pixel, the closest one is kept.
    sort_indices = np.argsort(z_valid)[::-1]
    u_sorted, v_sorted, z_sorted = u_valid[sort_indices], v_valid[sort_indices], z_valid[sort_indices]

    new_depth_image[v_sorted, u_sorted] = z_sorted

    # Convert back to millimeters and uint16 for saving
    new_depth_image_mm = (new_depth_image * depth_scale).astype(np.uint16)

    return new_depth_image_mm


def lidar_to_depth_image(lidar_file_path, T_laser_to_color_cam, color_intr, color_img_shape, depth_scale=1000.0):
    """
    Projects a LiDAR point cloud to a color camera's view to generate a depth map.

    Args:
        lidar_file_path (str): Path to the LiDAR .ply file.
        T_laser_to_color_cam (np.ndarray): 4x4 transform from LiDAR to color camera frame.
        color_intr (np.ndarray): 3x3 intrinsics of the color camera.
        color_img_shape (tuple): The (height, width) of the color image.
        depth_scale (float): Factor to convert meters to the final depth unit (e.g., 1000.0 for mm).

    Returns:
        np.ndarray: A new depth map aligned with the color camera's view, with values in mm (uint16).
    """
    h, w = color_img_shape[:2]

    # 1. Load and transform LiDAR point cloud
    lidar_pcd = o3d.io.read_point_cloud(lidar_file_path)
    lidar_pcd.transform(T_laser_to_color_cam)
    points_in_cam = np.asarray(lidar_pcd.points)

    # 2. Project points into the color image plane
    fx, fy = color_intr[0, 0], color_intr[1, 1]
    cx, cy = color_intr[0, 2], color_intr[1, 2]

    z_c = points_in_cam[:, 2]
    u_c = (points_in_cam[:, 0] * fx / z_c) + cx
    v_c = (points_in_cam[:, 1] * fy / z_c) + cy

    # 3. Filter points outside the view frustum
    valid_proj_mask = (z_c > 0.1) & (u_c >= 0) & (u_c < w) & (v_c >= 0) & (v_c < h)
    u_valid = u_c[valid_proj_mask].astype(int)
    v_valid = v_c[valid_proj_mask].astype(int)
    z_valid = z_c[valid_proj_mask]  # Depth is in meters

    # 4. Create new depth image, handling occlusions
    new_depth_image = np.zeros((h, w), dtype=np.float32)
    sort_indices = np.argsort(z_valid)[::-1]  # Sort descending to draw far points first
    u_sorted, v_sorted, z_sorted = u_valid[sort_indices], v_valid[sort_indices], z_valid[sort_indices]

    new_depth_image[v_sorted, u_sorted] = z_sorted

    # 5. Convert to millimeters and uint16 for saving
    new_depth_image_mm = (new_depth_image * depth_scale).astype(np.uint16)

    return new_depth_image_mm


def generate_point_cloud(rgb_path, depth_path, intrinsic_path, pose_path, depth_scale=1000.0, depth_trunc=3.0):
    """
    根据 RGB 图像、深度图、内参和外参生成 3D 点云。

    参数:
        rgb_path (str): RGB 图像文件路径。
        depth_path (str): 深度图文件路径。
        intrinsic_path (str): 内参矩阵文件路径。
        pose_path (str): 外参矩阵（相机到世界坐标）文件路径。
        depth_scale (float): 深度图缩放因子，用于将像素值转换为米。
        depth_trunc (float): 最大有效深度值（单位：米）。

    返回:
        open3d.geometry.PointCloud: 生成的 3D 点云。
    """
    # 加载 RGB 图像
    color_img = cv2.imread(rgb_path)
    if color_img is None:
        raise FileNotFoundError(f"无法加载 RGB 图像: {rgb_path}")

    # 加载深度图
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        raise FileNotFoundError(f"无法加载深度图: {depth_path}")

    # 归一化深度图（如果使用 uint16，则除以 depth_scale）
    depth_img = depth_img.astype(np.float32) / depth_scale

    # 应用深度截断
    depth_img[depth_img > depth_trunc] = 0

    # 加载内参矩阵
    intrinsic = np.loadtxt(intrinsic_path)
    if intrinsic.shape != (3, 3):
        raise ValueError(f"无效的内参矩阵形状: {intrinsic.shape}")

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    # 加载相机到世界的变换矩阵
    pose_w2c = np.loadtxt(pose_path)
    if pose_w2c.shape != (4, 4):
        raise ValueError(f"无效的外参矩阵形状: {pose_w2c.shape}")

    # 相机到世界坐标的变换矩阵
    T_c2w = np.linalg.inv(pose_w2c)

    # 创建点云
    h, w = depth_img.shape
    v, u = np.indices((h, w))
    z = depth_img[v, u]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # 过滤无效点
    valid_mask = z > 0
    points_cam = np.stack([x[valid_mask], y[valid_mask], z[valid_mask]], axis=-1)

    # 将点从相机坐标系转换到世界坐标系
    ones = np.ones((points_cam.shape[0], 1))
    points_homogeneous = np.hstack([points_cam, ones])
    points_world = (T_c2w @ points_homogeneous.T).T[:, :3]

    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)

    # 提取颜色信息
    colors = color_img[valid_mask]
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    return pcd


# =================================================================================
# Main Processing Function
# =================================================================================

def calculate_pose(row):
    T_odom_2_base = np.array(json.loads(row['odom_data']['homogeneous_matrix'])).reshape(4, 4)
    tf_data = row['tf_map2odom_data']
    T_map_2_odom = pose_to_homogeneous_matrix(
        tf_data['translation_x'], tf_data['translation_y'], tf_data['translation_z'],
        tf_data['rotation_x'], tf_data['rotation_y'], tf_data['rotation_z'], tf_data['rotation_w']
    ).reshape(4, 4)

    rotation_matrix_1 = np.array(
        [[0, 1, 0, 0],
         [-1, 0, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]
    )
    rotation_matrix_2 = np.array(
        [[0, 0, 1, 0],
         [0, 1, 0, 0],
         [-1, 0, 0, 0],
         [0, 0, 0, 1]]
    )
    rotation_matrix = rotation_matrix_2 @ rotation_matrix_1

    # Calculate camera-to-world transform (extrinsics)
    T_c2w = T_map_2_odom @ T_odom_2_base @ rotation_matrix

    return T_c2w


# =================================================================================
# Main Processing Function
# =================================================================================

def process_wheeltec_to_labelmaker(
        scan_dir,
        target_dir,
        frame_interval=1,
        sdf_trunc=0.04,
        voxel_length=0.008,
        depth_trunc=3.0,
        build_mesh=True,
        only_build_mesh=False
):
    """
    Converts a Wheeltec dataset recording to the LabelMaker format.
    """
    scan_path = Path(scan_dir)
    target_path = Path(target_dir)

    # 1. Define input directories from scan_dir
    image_dir = scan_path / "images"
    depth_dir = scan_path / "depth_point_clouds"  # Contains .npy depth maps
    lidar_dir = scan_path / "lidar_point_clouds"
    odom_file = scan_path / "odom_combined/odom_combined.csv"
    tf_file = scan_path / "tf_data/tf_map_to_odom_combined.csv"

    for p in [image_dir, depth_dir, odom_file, tf_file]:
        if not p.exists():
            raise FileNotFoundError(f"Required input not found: {p}")

    # 2. Synchronize data
    # TODO 还可以根据激光雷达来进行匹配
    matched_df = match_all_sensors_by_image(
        image_dir=str(image_dir),
        depth_dir=str(depth_dir),
        lidar_dir=str(lidar_dir),
        odom_file=str(odom_file),
        tf_map_to_odom_file=str(tf_file)
    )

    # 3. Apply frame sampling
    if frame_interval > 1:
        print(f"Sampling 1 every {frame_interval} frames...")
        matched_df = matched_df.iloc[::frame_interval].reset_index(drop=True)
        print(f"Resulting in {len(matched_df)} frames to process.")

    # 4. Setup target directory
    # if target_path.exists():
    #     print(f"Target directory {target_path} exists. Removing it.")
    #     shutil.rmtree(target_path)

    print(f"Creating LabelMaker structure at {target_path}")
    (target_path / 'color').mkdir(parents=True, exist_ok=True)
    (target_path / 'depth_from_camera').mkdir(exist_ok=True)
    (target_path / 'depth_from_lidar').mkdir(exist_ok=True)
    (target_path / 'intrinsic').mkdir(exist_ok=True)
    (target_path / 'pose').mkdir(exist_ok=True)

    if only_build_mesh:
        print("Fusing RGBD images into TSDF Volmue...")
        fuse_mesh(
            scan_dir=target_dir,
            sdf_trunc=sdf_trunc,
            voxel_length=voxel_length,
            depth_trunc=depth_trunc,
            depth_scale=1000.0,
        )  # depth_scale is a fixed value in ARKitScene, no need to pass an argument in cli
        print("Fusion finished! Saving to file as {}".format(os.path.join(target_dir, 'mesh.ply')))
        return

    # depth_from_depth_camera = True
    # depth_from_lidar = False

    # depth_source = 'depth_camera'  #
    # depth_source = 'lidar'  #
    depth_source = 'depth_camera_and_lidar'

    # 5. Process each frame
    correspondence_list = []
    for idx, row in tqdm(matched_df.iterrows(), total=len(matched_df), desc="Processing frames"):
        frame_id = f"{idx:06d}"

        # --- Process and Save Color Image ---
        original_color_path = row['image_file']
        target_color_path = target_path / 'color' / f"{frame_id}.jpg"
        color_img = cv2.imread(original_color_path)
        cv2.imwrite(str(target_color_path), color_img)

        # --- Process and Save Depth Map ---

        if depth_source == 'depth_camera':
            target_depth_path = target_path / 'depth_from_camera' / f"{frame_id}.png"
            original_depth_path = row['depth_file']
            depth_map_raw = np.load(original_depth_path)
            final_depth_map = reproject_depth_to_color_view(
                depth_map=depth_map_raw,
                depth_intr=depth_camera_intrinsic_info,
                color_intr=color_camera_intrinsic_info,
                depth_to_color_extr=T_depth_2_camera_color_frame,
                color_img_shape=color_img.shape,
            )
            cv2.imwrite(str(target_depth_path), final_depth_map)
        elif depth_source == 'lidar':
            target_depth_path = target_path / 'depth_from_lidar' / f"{frame_id}.png"
            original_depth_path = row['lidar_file']
            final_depth_map = lidar_to_depth_image(
                lidar_file_path=original_depth_path,
                T_laser_to_color_cam=T_laser_2_camera_color,
                color_intr=color_camera_intrinsic_info,
                color_img_shape=color_img.shape
            )
            cv2.imwrite(str(target_depth_path), final_depth_map)
        elif depth_source == 'depth_camera_and_lidar':
            ############
            target_depth_path = target_path / 'depth_from_camera' / f"{frame_id}.png"
            original_depth_path = row['depth_file']
            depth_map_raw = np.load(original_depth_path)
            final_depth_map = reproject_depth_to_color_view(
                depth_map=depth_map_raw,
                depth_intr=depth_camera_intrinsic_info,
                color_intr=color_camera_intrinsic_info,
                depth_to_color_extr=T_depth_2_camera_color_frame,
                color_img_shape=color_img.shape,
            )
            cv2.imwrite(str(target_depth_path), final_depth_map)
            #############
            target_depth_path = target_path / 'depth_from_lidar' / f"{frame_id}.png"
            original_depth_path = row['lidar_file']
            final_depth_map = lidar_to_depth_image(
                lidar_file_path=original_depth_path,
                T_laser_to_color_cam=T_laser_2_camera_color,
                color_intr=color_camera_intrinsic_info,
                color_img_shape=color_img.shape
            )
            cv2.imwrite(str(target_depth_path), final_depth_map)
        else:
            raise ValueError(f"Unsupported depth source: {depth_source}")

        # cv2.imwrite(str(target_depth_path), final_depth_map)

        # --- Process and Save Intrinsics ---
        target_intrinsic_path = target_path / 'intrinsic' / f"{frame_id}.txt"
        np.savetxt(target_intrinsic_path, color_camera_intrinsic_info, fmt='%.8f')

        # --- Calculate and Save Pose ---
        target_pose_path = target_path / 'pose' / f"{frame_id}.txt"
        pose = calculate_pose(row)
        np.savetxt(target_pose_path, pose, fmt='%.8f')

        if debug:
            pcd = generate_point_cloud(target_color_path, target_depth_path, target_intrinsic_path, target_pose_path)
            o3d.io.write_point_cloud(os.path.join(target_dir, f"{frame_id}_output.ply"), pcd)

        # --- Record correspondence ---
        correspondence_list.append({
            'frame_id': frame_id,
            'original_timestamp': row['timestamp'],
            'original_color_path': original_color_path,
            'original_depth_path': original_depth_path,
            'depth_source': depth_source
        })

    # 6. Save correspondence JSON file
    corr_path = target_path / 'correspondence.json'
    print(f"Saving correspondence file to {corr_path}")
    with open(corr_path, 'w') as f:
        json.dump(correspondence_list, f, indent=4)

    print("Conversion to LabelMaker format complete.")

    # 7. Optionally generate and save the fused mesh
    if build_mesh:
        print("Fusing RGBD images into TSDF Volmue...")
        fuse_mesh(
            scan_dir=target_dir,
            sdf_trunc=sdf_trunc,
            voxel_length=voxel_length,
            depth_trunc=depth_trunc,
            depth_scale=1000.0,
        )  # depth_scale is a fixed value in ARKitScene, no need to pass an argument in cli
        print("Fusion finished! Saving to file as {}".format(os.path.join(target_dir, 'mesh.ply')))


# =================================================================================
# Argparse and Main Execution
# =================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert a Wheeltec dataset recording to the LabelMaker format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "scan_dir",
        type=str,
        help="Path to the root directory of the Wheeltec recording (e.g., 'record_20250711_1540')."
    )
    parser.add_argument(
        "target_dir",
        type=str,
        help="Path to the output directory where the LabelMaker structure will be created."
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=5,
        help="The interval for frame sampling. '1' means all frames, '5' means every 5th frame."
    )

    parser.add_argument("--sdf_trunc", type=float, default=0.04)
    parser.add_argument("--voxel_length", type=float, default=0.01)
    parser.add_argument("--depth_trunc", type=float, default=4.0)

    args = parser.parse_args()

    process_wheeltec_to_labelmaker(
        scan_dir=args.scan_dir,
        target_dir=args.target_dir,
        frame_interval=args.frame_interval,
        sdf_trunc=args.sdf_trunc,
        voxel_length=args.voxel_length,
        depth_trunc=args.depth_trunc,
        build_mesh=True,
        only_build_mesh=True
        # build_mesh=False,
    )


if __name__ == "__main__":
    main()
