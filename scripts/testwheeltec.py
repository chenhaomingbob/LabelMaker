import os
import numpy as np
import open3d as o3d
import pandas as pd
from bisect import bisect_left
import json
from scipy.spatial.transform import Rotation as R
from numba import njit
from pathlib import Path


# base_footprint -> laser
# base_footprint -> camera_link -> camera_depth_frame -> camera_color_frame


def quaternion_to_rotation_matrix(q):
    x, y, z, w = q
    R = np.array([
        [1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x ** 2 - 2 * y ** 2]
    ])
    return R


def create_transform_matrix(t, q):
    # 创建变换矩阵
    transform = np.eye(4)
    transform[:3, :3] = quaternion_to_rotation_matrix(q)
    transform[:3, 3] = t

    return transform


depth_camera_intrinsic_info = np.array(
    [
        [475.4315490722656, 0.0, 324.42584228515625],
        [0.0, 475.4315490722656, 197.35276794433594],
        [0.0, 0.0, 1.0]
    ]
)
color_camera_intrinsic_info = np.array(
    [
        [450.8250732421875, 0.0, 334.319580078125],
        [0.0, 450.8250732421875, 247.18316650390625],
        [0.0, 0.0, 1.0]
    ]
)

# depth frame to color frame
# t = np.array([-0.000447075, 0.010095531, 0.0000743030980229377])  # 单位是m?
# q = [-0.000136242, 0.001019766, 0.004025249, 0.999991357]
# R = quaternion_to_rotation_matrix(q)
# depth_2_color_Transform = np.eye(4)
# depth_2_color_Transform[:3, :3] = R
# depth_2_color_Transform[:3, 3] = t


# base_footprint -> camera_link -> camera_depth_frame -> camera_color_frame
T_base_footprint_2_camera_link = create_transform_matrix(
    t=np.array([0.32039, 0.00164, 0.13208]),
    q=np.array([0.0, 0.0, 0.0, 1])
)  # 将相机坐标系下的点变换到base_footprint坐标系
T_camera_link_2_camera_depth_frame = create_transform_matrix(
    t=np.array([0.0, 0.0, 0.0]),
    q=np.array([0.0, 0.0, 0.0, 1])
)
T_depth_2_camera_color_frame = create_transform_matrix(
    t=np.array([-0.000447075, 0.010095531, 0.0000743030980229377]),
    q=np.array([-0.000136242, 0.001019766, 0.004025249, 0.999991357])
)
T_base_footprint_2_camera_color = T_base_footprint_2_camera_link @ T_camera_link_2_camera_depth_frame @ T_depth_2_camera_color_frame
T_base_footprint_2_camera_depth = T_base_footprint_2_camera_link @ T_camera_link_2_camera_depth_frame

# base_footprint -> laser
T_base_footprint_2_laser = create_transform_matrix(
    t=np.array([0.09039, 0.05053, 0.32632]),
    q=np.array([0.0, 0.0, -0.706825181105366, 0.707388269167199])
)
#
# T_laser_2_base = np.linalg.inv(T_base_footprint_2_laser)
# T_laser_2_camera_color = T_laser_2_base @ T_base_footprint_2_camera_color
# 正确的静态变换链：Lidar -> base -> camera
T_laser_2_camera_color = np.linalg.inv(T_base_footprint_2_camera_color) @ T_base_footprint_2_laser

import matplotlib.pyplot as plt


def visualize_lidar_projection_with_depth(matched_df,
                                          color_camera_intrinsic,
                                          T_laser_2_color,
                                          output_dir='./'
                                          ):
    """
    可视化Lidar点云到图像的投影，并用颜色表示深度。

    Args:
        matched_df (pd.DataFrame): 匹配好的传感器数据DataFrame。
        color_camera_intrinsic (np.ndarray): 3x3 的彩色相机内参矩阵。
        T_laser_2_color (np.ndarray): 4x4 的从激光雷达坐标系到彩色相机光学坐标系的静态变换矩阵。
        output_dir (str): 输出可视化结果图像的目录。
    """
    print("开始生成Lidar投影深度可视化图像...")
    from tqdm import tqdm

    # 确保输出子目录存在
    vis_output_dir = Path(output_dir) / "projection_visualization"
    vis_output_dir.mkdir(exist_ok=True)

    for _, row in tqdm(matched_df.iterrows(), total=len(matched_df)):
        # 1. 加载数据
        lidar_pcd = o3d.io.read_point_cloud(row['lidar_file'])
        image_path = row['image_file']

        # 使用 matplotlib 加载图像，以便后续绘制
        try:
            image_np = plt.imread(image_path)
        except FileNotFoundError:
            print(f"警告：找不到图像文件 {image_path}，跳过此帧。")
            continue

        img_h, img_w, _ = image_np.shape

        # points_in_camera_frame = np.asarray(lidar_pcd.points) @ T_laser_2_color[:3, :3].T + T_laser_2_color[:3, 3]
        # points_in_camera_frame = lidar_pcd.transform(T_laser_2_color)
        # points_in_camera_frame = lidar_pcd.transform(T_laser_2_color)
        lidar_pcd = lidar_pcd.transform(T_laser_2_color)

        # 绕 Z 轴顺时针旋转90度。
        rotation_matrix_1 = np.array(
            [[0, 1, 0, 0],
             [-1, 0, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]]
        )
        #  Y 轴顺时针旋转90度。
        rotation_matrix_2 = np.array(
            [[0, 0, 1, 0],
             [0, 1, 0, 0],
             [-1, 0, 0, 0],
             [0, 0, 0, 1]]
        )
        rotation_matrix = rotation_matrix_2 @ rotation_matrix_1
        points_in_camera_frame = lidar_pcd.transform(np.linalg.inv(rotation_matrix))

        points_in_camera_frame = np.asarray(points_in_camera_frame.points)

        # 2. 将Lidar点变换到相机坐标系

        # 3. 投影到图像平面并过滤
        # 过滤掉在相机后面或非常近的点（例如距离小于0.1米）
        valid_idx_z = points_in_camera_frame[:, 2] > 0.1

        # 投影计算
        fx, fy = color_camera_intrinsic[0, 0], color_camera_intrinsic[1, 1]
        cx, cy = color_camera_intrinsic[0, 2], color_camera_intrinsic[1, 2]

        u = (points_in_camera_frame[:, 0] * fx / points_in_camera_frame[:, 2] + cx)
        v = (points_in_camera_frame[:, 1] * fy / points_in_camera_frame[:, 2] + cy)

        # 过滤掉超出图像边界的点
        valid_idx_u = (u >= 0) & (u < img_w)
        valid_idx_v = (v >= 0) & (v < img_h)

        valid_mask = valid_idx_z & valid_idx_u & valid_idx_v

        # 获取所有有效投影点的像素坐标和深度
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        depth_valid = points_in_camera_frame[valid_mask, 2]

        if len(depth_valid) == 0:
            print(f"警告：在时间戳 {row['timestamp']} 没有有效的Lidar点投影到图像上。")
            continue

        # 4. 创建可视化图像
        fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
        ax.imshow(image_np)

        # 使用散点图绘制投影点，颜色由深度决定
        scatter = ax.scatter(u_valid, v_valid, c=depth_valid, cmap='jet', s=1.5, alpha=0.7)

        # 添加颜色条作为图例
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Depth (meters)')

        ax.set_title(f"Lidar Projection with Depth, Timestamp: {row['timestamp']}")
        ax.axis('off')  # 关闭坐标轴

        # 5. 保存图像
        output_filename = vis_output_dir / f"projection_{row['timestamp']}.png"
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
        plt.close(fig)  # 关闭图像以释放内存

    print(f"可视化图像已保存至: {vis_output_dir}")


def pose_to_homogeneous_matrix(tran_x, tran_y, tran_z, rot_x, rot_y, rot_z, rot_w):
    """将位姿转换为齐次变换矩阵"""
    position = np.array([tran_x, tran_y, tran_z])
    rotation_matrix = quaternion_to_rotation_matrix([rot_x, rot_y, rot_z, rot_w])
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotation_matrix
    homogeneous_matrix[:3, 3] = position
    return homogeneous_matrix


def find_closest_timestamp(target, timestamps):
    """找到最接近目标时间戳的索引"""
    pos = bisect_left(timestamps, target)
    if pos == 0:
        return 0
    if pos == len(timestamps):
        return pos - 1
    before = timestamps[pos - 1]
    after = timestamps[pos]
    if after - target < target - before:
        return pos
    else:
        return pos - 1


def match_depth_files(csv_path, npy_files):
    """
    主函数：匹配CSV中的时间戳与.npy文件

    参数:
        csv_path: CSV文件路径
        npy_dir: 存放.npy文件的目录

    返回:
        包含匹配结果的DataFrame
    """
    # 1. 读取CSV文件
    df = pd.read_csv(csv_path)

    # 2. 获取所有.npy文件的时间戳
    npy_timestamps = []

    for file in npy_files:
        try:
            # 从文件名提取时间戳 (假设文件名格式为"时间戳.npy")
            timestamp = float(os.path.splitext(os.path.basename(file))[0])
            npy_timestamps.append((timestamp, file))
        except ValueError:
            continue  # 跳过不符合命名规则的文件

    if not npy_timestamps:
        raise ValueError("未找到有效的.npy文件")

    # 按时间戳排序
    npy_timestamps.sort()
    timestamps_sorted = [t[0] for t in npy_timestamps]

    # 3. 为每个CSV行找到最接近的.npy文件
    matched_files = []
    time_diffs = []

    for ts in df['timestamp']:
        idx = find_closest_timestamp(ts, timestamps_sorted)
        closest_ts, closest_file = npy_timestamps[idx]
        matched_files.append(closest_file)
        time_diffs.append(abs(ts - closest_ts))

    # 4. 将匹配结果添加到DataFrame
    df['matched_depth_file'] = matched_files
    df['time_difference'] = time_diffs

    return df


def depth_to_pointcloud(depth_map, intrinsic_matrix, homogeneous_matrix=None):
    """
    将深度图转换为点云（可选择应用齐次变换矩阵）

    Args:
        depth_map (np.ndarray): (H, W) 深度图
        intrinsic_matrix (np.ndarray): 3×3 相机内参矩阵
        homogeneous_matrix (np.ndarray, optional): 4×4 齐次变换矩阵（默认 None）

    Returns:
        np.ndarray: (N, 3) 点云坐标
    """
    H, W = depth_map.shape
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    # 生成像素坐标网格
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.astype(float)
    v = v.astype(float)

    # 计算点云坐标（相机坐标系）
    z = depth_map
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # 转换为 (N, 3) 点云
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # 移除无效点（深度为 0 或 NaN）
    valid_mask = (z.flatten() > 0) & (~np.isnan(z.flatten()))
    points = points[valid_mask]

    # 应用齐次变换矩阵（如果提供）
    if homogeneous_matrix is not None:
        # 转换为齐次坐标 (N, 4)
        points_hom = np.hstack([points, np.ones((len(points), 1))])
        # 应用变换
        points_transformed = (homogeneous_matrix @ points_hom.T).T[:, :3]
        return points_transformed
    else:
        return points


def generate_and_merge_pointclouds(result_df, intrinsic_matrix, depth_dir):
    """
    遍历 DataFrame，生成所有点云并合并

    Args:
        result_df (pd.DataFrame): 包含 matched_depth_file 和 homogeneous_matrix
        intrinsic_matrix (np.ndarray): 3×3 相机内参矩阵

    Returns:
        np.ndarray: (N, 3) 合并后的全局点云
    """
    pointcloud_dict = {}  # 存储所有点云

    interval = 10
    for idx, row in result_df.iterrows():
        # if idx % interval != 0:
        #     continue
        # if idx>10:
        #     break

        depth_file = row["matched_depth_file"]
        timestamp = os.path.splitext(os.path.basename(depth_file))[0]
        homogeneous_matrix = np.array(json.loads(row["homogeneous_matrix"])).reshape(4, 4)  # 假设是 4×4 矩阵
        # homogeneous_matrix[0, 3] = 0
        # homogeneous_matrix[1, 3], homogeneous_matrix[2, 3] = homogeneous_matrix[2, 3], homogeneous_matrix[1, 3]
        # homogeneous_matrix[1, 3] = 0
        # homogeneous_matrix[2, 3] = 0

        # 像素 -> 相机 -> 世界
        rotation_matrix_1 = np.array(
            [[0, 1, 0, 0],
             [-1, 0, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]]
        )  # 绕z轴顺时针旋转270度 (从z轴往下看)
        rotation_matrix_2 = np.array(
            [[0, 0, 1, 0],
             [0, 1, 0, 0],
             [-1, 0, 0, 0],
             [0, 0, 0, 1]]
        )  # 绕y轴顺时针旋转90度 (从y轴往下看)
        rotation_matrix = rotation_matrix_2 @ rotation_matrix_1  # 先rotation_matrix_1 再  rotation_matrix_2
        homogeneous_matrix = homogeneous_matrix @ rotation_matrix
        # if depth_file == '2477.59201758.npy':
        # if depth_file == '2589.288507709.npy':
        # if depth_file == '2551.609003269.npy':
        if depth_file == '2477.59201758.npy' or depth_file == '2551.609003269.npy':
            pass

            # print(homogeneous_matrix)
            # homogeneous_matrix[0,3]=0
            # homogeneous_matrix[1,3 ] = 0
            # homogeneous_matrix[2,3 ] = 0
            # print(homogeneous_matrix)
            # print(row["position_x"], row["position_y"], row["position_z"])
        else:
            continue

        # 加载深度图
        depth_map = np.load(os.path.join(depth_dir, depth_file))

        print(np.unique(depth_map))

        # 转换为点云（应用齐次变换）
        points = depth_to_pointcloud(
            depth_map / 1000.0,
            intrinsic_matrix,
            homogeneous_matrix
        )

        # 添加到全局点云列表
        pointcloud_dict[timestamp] = {
            "points": points,
            "homogeneous_matrix": homogeneous_matrix
        }

        # source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # print(len(all_points))
    # 合并所有点云

    # if len()

    # merged_pointcloud = np.vstack(all_points)
    return pointcloud_dict


# def match_all_sensors(image_dir, depth_dir, lidar_dir, odom_file, tf_map_to_odom_file):
#     """
#     匹配时间戳最近的图像、odom、深度图和tf_map_to_odom数据
#
#     Args:
#         image_dir: 图像文件目录(.jpg/.png)
#         depth_dir: 深度图文件目录(.npy)
#         odom_file: odom数据的CSV文件路径
#         tf_map_to_odom_file: tf_map_to_odom数据的CSV文件路径
#
#     Returns:
#         pd.DataFrame: 包含匹配结果的DataFrame
#     """
#     # 获取所有图像文件
#     image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
#     # 获取所有深度图文件
#     depth_files = [f for f in os.listdir(depth_dir) if f.endswith('.npy')]
#
#     # 读取odom数据
#     odom_df = pd.read_csv(odom_file)
#     # 读取tf_map_to_odom数据
#     tf_df = pd.read_csv(tf_map_to_odom_file)
#
#     # 提取所有时间戳
#     image_timestamps = [float(os.path.splitext(f)[0]) for f in image_files]
#     depth_timestamps = [float(os.path.splitext(f)[0]) for f in depth_files]
#     odom_timestamps = odom_df['timestamp'].values
#     tf_timestamps = tf_df['timestamp'].values
#
#     # 确保所有时间戳已排序
#     image_timestamps.sort()
#     depth_timestamps.sort()
#     tf_timestamps.sort()
#
#     # 创建匹配结果的DataFrame
#     matched_data = []
#
#     # 遍历odom数据作为基准
#     for odom_ts in odom_timestamps:
#         # 找到最接近的图像
#         image_idx = find_closest_timestamp(odom_ts, image_timestamps)
#         image_ts = image_timestamps[image_idx]
#         image_file = f"{image_ts}.jpg" if f"{image_ts}.jpg" in image_files else f"{image_ts}.png"
#
#         # 找到最接近的深度图
#         depth_idx = find_closest_timestamp(odom_ts, depth_timestamps)
#         depth_ts = depth_timestamps[depth_idx]
#         depth_file = f"{depth_ts}.npy"
#
#         # 找到最接近的tf_map_to_odom数据
#         tf_idx = find_closest_timestamp(odom_ts, tf_timestamps)
#         tf_ts = tf_timestamps[tf_idx]
#
#         # 获取odom行数据
#         odom_row = odom_df[odom_df['timestamp'] == odom_ts].iloc[0]
#         # 获取tf_map_to_odom行数据
#         tf_row = tf_df[tf_df['timestamp'] == tf_ts].iloc[0]
#
#         # 添加到匹配结果
#         matched_data.append({
#             'timestamp': odom_ts,
#             'image_timestamp': image_ts,
#             'depth_timestamp': depth_ts,
#             'tf_timestamp': tf_ts,
#             'image_file': os.path.join(image_dir, image_file),
#             'depth_file': os.path.join(depth_dir, depth_file),
#             'odom_data': odom_row.to_dict(),
#             'tf_map2odom_data': tf_row.to_dict(),  # 假定map
#             'image_time_diff': abs(odom_ts - image_ts),
#             'depth_time_diff': abs(odom_ts - depth_ts),
#             'tf_time_diff': abs(odom_ts - tf_ts)
#         })
#
#     # 转换为DataFrame
#     matched_df = pd.DataFrame(matched_data)
#
#     return matched_df


def match_all_sensors(image_dir, depth_dir, lidar_dir, odom_file, tf_map_to_odom_file, max_time_diff_sec=0.05):
    """
    以图像时间戳为基准，匹配时间戳最近的odom, 深度图, 激光雷达和tf_map_to_odom数据。

    Args:
        image_dir: 图像文件目录(.jpg/.png)
        depth_dir: 深度图文件目录(.npy)
        lidar_dir: 激光雷达数据文件目录(.npy)
        odom_file: odom数据的CSV文件路径
        tf_map_to_odom_file: tf_map_to_odom数据的CSV文件路径

    Returns:
        pd.DataFrame: 包含匹配结果的DataFrame
    """
    # 1. 获取所有传感器的文件和数据
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png'))]
    depth_files = [f for f in os.listdir(depth_dir) if f.endswith('.npy')]
    lidar_files = [f for f in os.listdir(lidar_dir) if f.endswith('.ply')]

    # 读取CSV数据
    odom_df = pd.read_csv(odom_file)
    tf_df = pd.read_csv(tf_map_to_odom_file)

    # 2. 提取并排序所有时间戳
    image_timestamps = sorted([float(os.path.splitext(f)[0]) for f in image_files])
    depth_timestamps = sorted([float(os.path.splitext(f)[0]) for f in depth_files])
    lidar_timestamps = sorted([float(os.path.splitext(f)[0]) for f in lidar_files])
    odom_timestamps = sorted(odom_df['timestamp'].unique())
    tf_timestamps = sorted(tf_df['timestamp'].unique())

    # 检查时间戳列表是否为空
    if not image_timestamps:
        raise ValueError("图像目录中未找到任何图像文件。")

    # 3. 创建匹配结果列表
    matched_data = []

    # 4. 以激光雷达时间戳为基准进行遍历和匹配
    ref_timestamps = lidar_timestamps
    for ref_ts in ref_timestamps:
        # 获取激光雷达文件名
        lidar_file = f"{ref_ts}.ply"

        # 找到最接近的图像
        image_idx = find_closest_timestamp(ref_ts, image_timestamps)
        image_ts = image_timestamps[image_idx]
        image_file = f"{image_ts}.jpg" if f"{image_ts}.jpg" in image_files else f"{image_ts}.png"

        # 找到最接近的深度图
        depth_idx = find_closest_timestamp(ref_ts, depth_timestamps)
        depth_ts = depth_timestamps[depth_idx]
        depth_file = f"{depth_ts}.npy"

        # 找到最接近的Odom数据
        odom_idx = find_closest_timestamp(ref_ts, odom_timestamps)
        odom_ts = odom_timestamps[odom_idx]
        odom_row = odom_df[odom_df['timestamp'] == odom_ts].iloc[0]

        # 找到最接近的tf_map_to_odom数据
        tf_idx = find_closest_timestamp(ref_ts, tf_timestamps)
        tf_ts = tf_timestamps[tf_idx]
        tf_row = tf_df[tf_df['timestamp'] == tf_ts].iloc[0]

        # --- 检查时间戳差异是否在阈值内（以lidar_ts为中心） ---
        if (abs(ref_ts - image_ts) > max_time_diff_sec or
                abs(ref_ts - depth_ts) > max_time_diff_sec or
                abs(ref_ts - odom_ts) > max_time_diff_sec or
                abs(ref_ts - tf_ts) > max_time_diff_sec):
            continue  # 如果任何一个时间差超限，则跳过当前激光雷达帧

        # 5. 将所有匹配结果添加到列表
        matched_data.append({
            'timestamp': ref_ts,  # 基准时间戳
            'lidar_file': os.path.join(lidar_dir, lidar_file),

            'image_timestamp': image_ts,
            'image_file': os.path.join(image_dir, image_file),
            'image_time_diff': abs(ref_ts - image_ts),

            'depth_timestamp': depth_ts,
            'depth_file': os.path.join(depth_dir, depth_file),
            'depth_time_diff': abs(ref_ts - depth_ts),

            'odom_timestamp': odom_ts,
            'odom_data': odom_row.to_dict(),
            'odom_time_diff': abs(ref_ts - odom_ts),

            'tf_timestamp': tf_ts,
            'tf_map2odom_data': tf_row.to_dict(),
            'tf_time_diff': abs(ref_ts - tf_ts)
        })

    # 6. 转换为DataFrame并返回
    matched_df = pd.DataFrame(matched_data)
    return matched_df


def generate_color_pointcloud(matched_df,
                              depth_intrinsic_matrix,
                              color_intrinsic_matrix,
                              depth_dir,
                              image_dir,
                              depth2color_matrix,
                              use_ICP_register=False,
                              output_dir='./'
                              ):
    """
    生成彩色点云，融合RGB图像和深度信息

    Args:
        matched_df: 匹配好的传感器数据DataFrame
        intrinsic_matrix: 相机内参矩阵
        depth_dir: 深度图目录
        image_dir: 图像目录

    Returns:
        o3d.geometry.PointCloud: 带颜色的点云对象
    """
    # 创建空的点云对象
    color_pointcloud = o3d.geometry.PointCloud()

    all_points = []
    all_colors = []
    all_homogeneous_matrix = []
    from tqdm import tqdm
    for _, row in tqdm(matched_df.iterrows()):
        # 加载深度图
        depth_file = os.path.join(depth_dir, os.path.basename(row['depth_file']))
        depth_map = np.load(depth_file)

        # 加载图像
        image_file = os.path.join(image_dir, os.path.basename(row['image_file']))
        image = o3d.io.read_image(image_file)
        image_array = np.asarray(image)

        # 获取变换矩阵 base -> world
        homogeneous_matrix = np.array(json.loads(row['odom_data']['homogeneous_matrix'])).reshape(4, 4)

        # map
        tf_map2odom_matrix = np.array(
            pose_to_homogeneous_matrix(
                row['tf_map2odom_data']['translation_x'],
                row['tf_map2odom_data']['translation_y'],
                row['tf_map2odom_data']['translation_z'],
                row['tf_map2odom_data']['rotation_x'],
                row['tf_map2odom_data']['rotation_y'],
                row['tf_map2odom_data']['rotation_z'],
                row['tf_map2odom_data']['rotation_w'],
            )
        ).reshape(4, 4)
        # tf_map2odom_matrix = np.eye(4).reshape(4, 4)

        # homogeneous_matrix = np.linalg.inv(homogeneous_matrix)
        # tf_map2odom_matrix = np.linalg.inv(tf_map2odom_matrix)

        # 旋转矩阵处理(保持与之前一致)
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
        # homogeneous_matrix = homogeneous_matrix @ rotation_matrix

        # 生成点云(不应用变换，我们先获取相机坐标系下的点)
        depth_points = depth_to_pointcloud(depth_map / 1000.0, depth_intrinsic_matrix)
        depth_points_hom = np.hstack([depth_points, np.ones((len(depth_points), 1))])

        color_points_transformed = (depth2color_matrix @ depth_points_hom.T).T

        z_points = color_points_transformed[:, 2]

        u_color = (color_intrinsic_matrix[0, 0] * color_points_transformed[:, 0] / color_points_transformed[:, 2] +
                   color_intrinsic_matrix[0, 2])
        v_color = (color_intrinsic_matrix[1, 1] * color_points_transformed[:, 1] / color_points_transformed[:, 2] +
                   color_intrinsic_matrix[1, 2])

        mask_1 = u_color > 0
        mask_2 = u_color < 640
        mask_3 = v_color > 0
        mask_4 = v_color < 480
        mask_5 = z_points > 0

        valid_mask = mask_1 & mask_2 & mask_3 & mask_4 & mask_5

        # 获取有效点的颜色(与depth_to_pointcloud相同的mask)
        u_color = u_color[valid_mask]
        v_color = v_color[valid_mask]
        colors = image_array[(v_color.flatten()).astype(int), (u_color.flatten()).astype(int)] / 255.0

        # 应用变换矩阵
        depth_points_hom = depth_points_hom[valid_mask]  # 保留有RGB对应的深度点

        depth_points_hom = (tf_map2odom_matrix @ homogeneous_matrix @ rotation_matrix @ depth_points_hom.T).T[:, :3]
        points_transformed = depth_points_hom[:, :3]

        all_points.append(points_transformed)
        all_colors.append(colors)
        all_homogeneous_matrix.append(homogeneous_matrix)

    global_pointcloud = o3d.geometry.PointCloud()
    global_pointcloud.points = o3d.utility.Vector3dVector(all_points[0])
    global_pointcloud.colors = o3d.utility.Vector3dVector(all_colors[0])
    voxel_size = 0.02
    global_pointcloud = global_pointcloud.voxel_down_sample(voxel_size)
    # ICP参数设置
    threshold = 0.02  # 距离阈值

    from collections import deque
    # 保留最近N个关键帧的点云
    recent_frames = deque(maxlen=20)  # 最多保留20帧
    for i in tqdm(range(1, len(all_points))):
        # 当前点云
        current_point_cloud = o3d.geometry.PointCloud()
        current_point_cloud.points = o3d.utility.Vector3dVector(all_points[i])
        current_point_cloud.colors = o3d.utility.Vector3dVector(all_colors[i])
        current_point_cloud = current_point_cloud.voxel_down_sample(voxel_size)

        if use_ICP_register:
            # 使用ICP算法进行配准
            reg_p2p = o3d.pipelines.registration.registration_icp(
                current_point_cloud, global_pointcloud, threshold, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            # 获取变换矩阵
            transformation_icp = reg_p2p.transformation

            # 应用变换矩阵
            current_point_cloud.transform(transformation_icp)

        # 合并点云
        global_pointcloud += current_point_cloud
        global_pointcloud = global_pointcloud.voxel_down_sample(voxel_size)
        print("i:", i)

    o3d.io.write_point_cloud(
        os.path.join(output_dir, "color_pcd_by_icp_aligned.ply" if use_ICP_register else "color_pcd.ply"),
        global_pointcloud
    )

    #################
    global_pointcloud, _, inverse_index_list = global_pointcloud.voxel_down_sample_and_trace(
        voxel_size=voxel_size, min_bound=global_pointcloud.get_min_bound(), max_bound=global_pointcloud.get_max_bound()
    )

    _, ind1 = global_pointcloud.remove_radius_outlier(nb_points=30, radius=0.05)
    # _, ind1 = global_pointcloud.remove_radius_outlier(nb_points=50, radius=0.08)
    global_pointcloud1 = global_pointcloud.select_by_index(ind1)
    o3d.io.write_point_cloud(
        os.path.join(output_dir,
                     "color_pcd_by_icp_aligned_remove_ro.ply" if use_ICP_register else "color_pcd_remove_ro.ply"),
        global_pointcloud1
    )

    _, ind2 = global_pointcloud.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
    global_pointcloud2 = global_pointcloud.select_by_index(ind2)
    o3d.io.write_point_cloud(
        os.path.join(
            output_dir,
            "color_pcd_by_icp_aligned_remove_so.ply" if use_ICP_register else "color_pcd_remove_so.ply"),
        global_pointcloud2
    )

    _, ind2 = global_pointcloud1.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    # _, ind2 = global_pointcloud1.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    global_pointcloud3 = global_pointcloud1.select_by_index(ind2)
    o3d.io.write_point_cloud(
        os.path.join(
            output_dir,
            "color_pcd_by_icp_aligned_remove_ro_so.ply" if use_ICP_register else "color_pcd_remove_ro_so.ply"),
        global_pointcloud3
    )

    #################

    # 合并所有点云和颜色
    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors)

    # 创建Open3D点云对象
    color_pointcloud.points = o3d.utility.Vector3dVector(merged_points)
    color_pointcloud.colors = o3d.utility.Vector3dVector(merged_colors)

    return color_pointcloud


def generate_lidar_color_pointcloud(matched_df,
                                    color_camera_intrinsic,
                                    T_laser_2_color,
                                    # T_base_2_color,
                                    # T_base_2_laser,
                                    use_ICP_register=False,
                                    output_dir='./'
                                    ):
    """
    根据Lidar点云和匹配的图像生成彩色点云。

    Args:
        matched_df (pd.DataFrame): 匹配好的传感器数据DataFrame，以Lidar为基准。
        color_camera_intrinsic (np.ndarray): 3x3 的彩色相机内参矩阵。
        T_base_2_color (np.ndarray): 4x4 的从基座(base_footprint)到彩色相机光学坐标系的变换矩阵。
        T_base_2_laser (np.ndarray): 4x4 的从基座(base_footprint)到激光雷达坐标系的变换矩阵。
        use_ICP_register (bool): 是否使用ICP进行点云配准。
        output_dir (str): 输出文件的目录。

    Returns:
        o3d.geometry.PointCloud: 拼接并着色后的全局Lidar点云。
    """
    all_colored_points = []
    from tqdm import tqdm

    # 计算从Lidar到相机光学坐标系的静态变换矩阵的逆，即从相机到Lidar
    # 我们需要的是 lidar -> base -> camera
    # T_laser_2_base = np.linalg.inv(T_base_2_laser)
    # T_color_2_base = np.linalg.inv(T_base_2_color)
    #
    # # 静态变换：从激光雷达坐标系变换到彩色相机光学坐标系
    # T_laser_2_color = T_color_2_base @ T_base_2_laser  # 这是错误的，应该是 T_cam_2_base 的逆 @ T_base_2_lidar
    # T_laser_2_color = np.linalg.inv(T_base_2_color) @ T_base_2_laser  # 正确的链式变换

    print("开始生成着色的Lidar点云...")
    for _, row in tqdm(matched_df.iterrows(), total=len(matched_df)):
        # 1. 加载数据
        lidar_pcd = o3d.io.read_point_cloud(row['lidar_file'])
        image_path = row['image_file']
        image = o3d.io.read_image(image_path)
        image_np = np.asarray(image)
        img_h, img_w, _ = image_np.shape

        o3d.io.write_point_cloud(os.path.join(output_dir, "world_pcd_0.ply"), lidar_pcd)

        points_in_camera_frame = lidar_pcd.transform(T_laser_2_color)  # world

        # world_pcd = lidar_pcd
        o3d.io.write_point_cloud(os.path.join(output_dir, "world_pcd_1.ply"), points_in_camera_frame)

        # 绕 Z 轴顺时针旋转90度。
        rotation_matrix_1 = np.array(
            [[0, 1, 0, 0],
             [-1, 0, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]]
        )
        #  Y 轴顺时针旋转90度。
        rotation_matrix_2 = np.array(
            [[0, 0, 1, 0],
             [0, 1, 0, 0],
             [-1, 0, 0, 0],
             [0, 0, 0, 1]]
        )
        rotation_matrix = rotation_matrix_2 @ rotation_matrix_1
        points_in_camera_frame = lidar_pcd.transform(np.linalg.inv(rotation_matrix))
        o3d.io.write_point_cloud(os.path.join(output_dir, "world_pcd_2.ply"), points_in_camera_frame)

        # 5. 投影到图像平面并着色
        # 过滤掉在相机后面的点
        points_in_camera_frame = np.asarray(points_in_camera_frame.points)
        valid_idx_z = points_in_camera_frame[:, 2] > 0

        # 投影计算
        fx, fy = color_camera_intrinsic[0, 0], color_camera_intrinsic[1, 1]
        cx, cy = color_camera_intrinsic[0, 2], color_camera_intrinsic[1, 2]

        u = (points_in_camera_frame[:, 0] * fx / points_in_camera_frame[:, 2] + cx).astype(int)
        v = (points_in_camera_frame[:, 1] * fy / points_in_camera_frame[:, 2] + cy).astype(int)

        # 过滤掉超出图像边界的点
        valid_idx_u = (u >= 0) & (u < img_w)
        valid_idx_v = (v >= 0) & (v < img_h)

        valid_mask = valid_idx_z & valid_idx_u & valid_idx_v

        # 获取有效点的颜色
        colors = image_np[v[valid_mask], u[valid_mask]] / 255.0

        colored_pcd_frame_colors = np.asarray(lidar_pcd.colors)
        colored_pcd_frame_colors[valid_mask, :] = image_np[v[valid_mask], u[valid_mask]] / 255.0

        # 创建一个新的点云对象用于存储当前帧的着色点
        colored_pcd_frame = o3d.geometry.PointCloud()
        colored_pcd_frame.points = o3d.utility.Vector3dVector(np.asarray(lidar_pcd.points)[valid_mask])
        colored_pcd_frame.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(os.path.join(output_dir, "world_pcd_3.ply"), colored_pcd_frame)

        colored_pcd_frame = o3d.geometry.PointCloud()
        colored_pcd_frame.points = o3d.utility.Vector3dVector(lidar_pcd.points)
        colored_pcd_frame.colors = o3d.utility.Vector3dVector(colored_pcd_frame_colors)
        o3d.io.write_point_cloud(os.path.join(output_dir, "world_pcd_4.ply"), colored_pcd_frame)

        all_colored_points.append(colored_pcd_frame)

    # 6. 合并所有点云
    print("合并所有着色点云...")
    global_colored_pcd = o3d.geometry.PointCloud()
    for pcd in all_colored_points:
        global_colored_pcd += pcd

    # 7. (可选) 后处理
    print("进行体素下采样...")
    voxel_size = 0.02  # 可以根据需要调整
    global_colored_pcd = global_colored_pcd.voxel_down_sample(voxel_size)

    # 保存未经过ICP的原始拼接结果
    o3d.io.write_point_cloud(os.path.join(output_dir, "lidar_colored_pcd_raw.ply"), global_colored_pcd)

    if use_ICP_register:
        # 这里的ICP逻辑可以根据需要实现，但对于已经有SLAM位姿的点云，通常不是必须的
        print("ICP配准被跳过（在此函数中未实现）。")
        pass

    # 移除离群点
    print("移除离群点...")
    pointcloud_ror, _ = global_colored_pcd.remove_radius_outlier(nb_points=20, radius=0.05)
    pointcloud_final, _ = pointcloud_ror.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)

    o3d.io.write_point_cloud(os.path.join(output_dir, "lidar_colored_pcd_final.ply"), pointcloud_final)

    return pointcloud_final


if __name__ == '__main__':
    base_dir = Path("/data1/chm/datasets/wheeltec/record_20250625")
    # 深度目录
    depth_dir = base_dir / "depth_point_clouds"
    # 激光目录
    lidar_dir = base_dir / "lidar_point_clouds"
    # 图像目录
    image_dir = base_dir / "images"
    # odom 文件
    odom_file = base_dir / "odom_combined/odom_combined.csv"
    #
    tf_map_to_odom_file = base_dir / "tf_data/tf_map_to_odom_combined.csv"

    output_dir = base_dir / "outputs_dirs"
    output_dir.mkdir(exist_ok=True)
    #######################################################
    # 执行匹配
    matched_df = match_all_sensors(
        image_dir=image_dir,
        depth_dir=depth_dir,
        lidar_dir=lidar_dir,
        odom_file=odom_file,
        tf_map_to_odom_file=tf_map_to_odom_file,
        max_time_diff_sec=0.05  # 传感器组中最慢频率周期的一半
    )

    matched_df = matched_df.iloc[::5]
    print(matched_df.head())
    print(len(matched_df))
    # 生成彩色点云
    # color_pcd = generate_color_pointcloud(
    #     matched_df,
    #     depth_camera_intrinsic_info,
    #     color_camera_intrinsic_info,
    #     depth_dir,
    #     image_dir,
    #     T_depth_2_camera_color_frame,  # 传递变换矩阵,
    #     use_ICP_register=False,
    #     output_dir=output_dir
    # )
    # o3d.io.write_point_cloud(os.path.join(output_dir, "color_pcd.ply"), color_pcd)

    # 2. 调用新函数生成可视化图像
    visualize_lidar_projection_with_depth(
        matched_df=matched_df,
        color_camera_intrinsic=color_camera_intrinsic_info,
        T_laser_2_color=T_laser_2_camera_color,
        output_dir=output_dir
    )

    # 2. 调用新函数生成着色的Lidar点云
    colored_lidar_pcd = generate_lidar_color_pointcloud(
        matched_df=matched_df,
        color_camera_intrinsic=color_camera_intrinsic_info,
        # T_base_2_color=T_base_footprint_2_camera_color,
        # T_base_2_laser=T_base_footprint_2_laser,
        T_laser_2_color=T_laser_2_camera_color,
        use_ICP_register=False,
        output_dir=output_dir
    )
    o3d.io.write_point_cloud(os.path.join(output_dir, "color_lidar_pcd.ply"), colored_lidar_pcd)
