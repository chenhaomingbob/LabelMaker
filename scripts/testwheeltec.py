import os
import numpy as np
import open3d as o3d
import pandas as pd
from bisect import bisect_left
import json
from scipy.spatial.transform import Rotation as R
from numba import njit


def quaternion_to_rotation_matrix(q):
    x, y, z, w = q
    R = np.array([
        [1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x ** 2 - 2 * y ** 2]
    ])
    return R


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
t = np.array([-0.000447075, 0.010095531, 0.0000743030980229377])  # 单位是m?
q = [-0.000136242, 0.001019766, 0.004025249, 0.999991357]
R = quaternion_to_rotation_matrix(q)
depth_2_color_Transform = np.eye(4)
depth_2_color_Transform[:3, :3] = R
depth_2_color_Transform[:3, 3] = t


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


def match_all_sensors(image_dir, depth_dir, odom_file, tf_map_to_odom_file):
    """
    匹配时间戳最近的图像、odom、深度图和tf_map_to_odom数据

    Args:
        image_dir: 图像文件目录(.jpg/.png)
        depth_dir: 深度图文件目录(.npy)
        odom_file: odom数据的CSV文件路径
        tf_map_to_odom_file: tf_map_to_odom数据的CSV文件路径

    Returns:
        pd.DataFrame: 包含匹配结果的DataFrame
    """
    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    # 获取所有深度图文件
    depth_files = [f for f in os.listdir(depth_dir) if f.endswith('.npy')]

    # 读取odom数据
    odom_df = pd.read_csv(odom_file)
    # 读取tf_map_to_odom数据
    tf_df = pd.read_csv(tf_map_to_odom_file)

    # 提取所有时间戳
    image_timestamps = [float(os.path.splitext(f)[0]) for f in image_files]
    depth_timestamps = [float(os.path.splitext(f)[0]) for f in depth_files]
    odom_timestamps = odom_df['timestamp'].values
    tf_timestamps = tf_df['timestamp'].values

    # 确保所有时间戳已排序
    image_timestamps.sort()
    depth_timestamps.sort()
    tf_timestamps.sort()

    # 创建匹配结果的DataFrame
    matched_data = []

    # 遍历odom数据作为基准
    for odom_ts in odom_timestamps:
        # 找到最接近的图像
        image_idx = find_closest_timestamp(odom_ts, image_timestamps)
        image_ts = image_timestamps[image_idx]
        image_file = f"{image_ts}.jpg" if f"{image_ts}.jpg" in image_files else f"{image_ts}.png"

        # 找到最接近的深度图
        depth_idx = find_closest_timestamp(odom_ts, depth_timestamps)
        depth_ts = depth_timestamps[depth_idx]
        depth_file = f"{depth_ts}.npy"

        # 找到最接近的tf_map_to_odom数据
        tf_idx = find_closest_timestamp(odom_ts, tf_timestamps)
        tf_ts = tf_timestamps[tf_idx]

        # 获取odom行数据
        odom_row = odom_df[odom_df['timestamp'] == odom_ts].iloc[0]
        # 获取tf_map_to_odom行数据
        tf_row = tf_df[tf_df['timestamp'] == tf_ts].iloc[0]

        # 添加到匹配结果
        matched_data.append({
            'timestamp': odom_ts,
            'image_timestamp': image_ts,
            'depth_timestamp': depth_ts,
            'tf_timestamp': tf_ts,
            'image_file': os.path.join(image_dir, image_file),
            'depth_file': os.path.join(depth_dir, depth_file),
            'odom_data': odom_row.to_dict(),
            'tf_map2odom_data': tf_row.to_dict(),
            'image_time_diff': abs(odom_ts - image_ts),
            'depth_time_diff': abs(odom_ts - depth_ts),
            'tf_time_diff': abs(odom_ts - tf_ts)
        })

    # 转换为DataFrame
    matched_df = pd.DataFrame(matched_data)

    return matched_df


def generate_color_pointcloud(matched_df,
                              depth_intrinsic_matrix,
                              color_intrinsic_matrix,
                              depth_dir,
                              image_dir,
                              depth2color_matrix,
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
    for _, row in matched_df.iterrows():
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

        # 转换后的点云
        # depth_points_transformed = (rotation_matrix @ points_hom.T).T  # 变换到z-up x-forward y-left
        # color_points_transformed = (depth2color_matrix @ depth_points_transformed.T).T  # （N,4）
        # color_points_transformed = (np.linalg.inv(rotation_matrix) @ color_points_transformed.T).T

        #
        # depth_points_transformed = (rotation_matrix @ points_hom.T).T  # 变换到z-up x-forward y-left
        color_points_transformed = (depth2color_matrix @ depth_points_hom.T).T
        # color_points_transformed = (np.linalg.inv(rotation_matrix) @ color_points_transformed.T).T

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
        # mask_1 = u_color > 200
        # mask_2 = u_color < 440
        # mask_3 = v_color > 100
        # mask_4 = v_color < 380
        # mask_5 = z_points > 0
        valid_mask = mask_1 & mask_2 & mask_3 & mask_4 & mask_5

        # 获取有效点的颜色(与depth_to_pointcloud相同的mask)
        u_color = u_color[valid_mask]
        v_color = v_color[valid_mask]
        colors = image_array[(v_color.flatten()).astype(int), (u_color.flatten()).astype(int)] / 255.0

        # 应用变换矩阵
        depth_points_hom = depth_points_hom[valid_mask]  # 保留有RGB对应的深度点
        # points_hom = depth_points
        # points_hom = np.hstack([points, np.ones((len(points), 1))])
        depth_points_hom = (tf_map2odom_matrix @ homogeneous_matrix @ rotation_matrix @ depth_points_hom.T).T[:, :3]
        # points_transformed = (homogeneous_matrix @ points_hom.T).T[:, :3]
        points_transformed = depth_points_hom[:, :3]
        print("*" * 50)
        print(row['timestamp'], row['image_file'], row['depth_file'], row['tf_timestamp'])
        print(tf_map2odom_matrix)
        print(homogeneous_matrix)
        print("*" * 50)

        # if len(all_points) > 0:
        #     print(1)
        #     threshold = 0.02  # 匹配距离阈值
        #     trans_init = np.identity(4)  # 初始变换矩阵(单位矩阵)
        #     source = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_transformed))
        #     target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_points[-1]))
        #     reg_p2p = o3d.pipelines.registration.registration_icp(
        #         source, target, threshold, trans_init,
        #         o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        #         o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
        #     source.transform(reg_p2p.transformation)
        #     points_transformed = np.array(source.points)

        all_points.append(points_transformed)
        all_colors.append(colors)
        all_homogeneous_matrix.append(homogeneous_matrix)

    global_pointcloud = o3d.geometry.PointCloud()
    global_pointcloud.points = o3d.utility.Vector3dVector(all_points[0])
    global_pointcloud.colors = o3d.utility.Vector3dVector(all_colors[0])
    # voxel_size = 0.02
    # global_pointcloud = global_pointcloud.voxel_down_sample(voxel_size)
    # ICP参数设置
    threshold = 0.02  # 距离阈值
    for i in range(1, len(all_points)):
        # 当前点云
        current_point_cloud = o3d.geometry.PointCloud()
        current_point_cloud.points = o3d.utility.Vector3dVector(all_points[i])
        current_point_cloud.colors = o3d.utility.Vector3dVector(all_colors[i])
        # current_point_cloud = current_point_cloud.voxel_down_sample(voxel_size)

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
        print("i:", i)

    o3d.io.write_point_cloud(os.path.join(output_dir, "color_pcd_by_icp_aligned.ply"), global_pointcloud)

    #################
    global_pointcloud, _, inverse_index_list = global_pointcloud.voxel_down_sample_and_trace(
        voxel_size=0.008, min_bound=global_pointcloud.get_min_bound(), max_bound=global_pointcloud.get_max_bound()
    )


    _, ind1 = global_pointcloud.remove_radius_outlier(nb_points=50, radius=0.08)
    global_pointcloud1 = global_pointcloud.select_by_index(ind1)
    o3d.io.write_point_cloud(os.path.join(output_dir, "color_pcd_by_icp_aligned_remove_ro.ply"), global_pointcloud1)

    _, ind2 = global_pointcloud.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    global_pointcloud2 = global_pointcloud.select_by_index(ind2)
    o3d.io.write_point_cloud(os.path.join(output_dir, "color_pcd_by_icp_aligned_remove_so.ply"), global_pointcloud2)

    _, ind2 = global_pointcloud1.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
    global_pointcloud3 = global_pointcloud1.select_by_index(ind2)
    o3d.io.write_point_cloud(os.path.join(output_dir, "color_pcd_by_icp_aligned_remove_ro_so.ply"), global_pointcloud3)

    #################

    # 合并所有点云和颜色
    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors)

    # 创建Open3D点云对象
    color_pointcloud.points = o3d.utility.Vector3dVector(merged_points)
    color_pointcloud.colors = o3d.utility.Vector3dVector(merged_colors)

    return color_pointcloud


# 深度图目录
depth_dir = "/data1/chm/datasets/wheeltec/record_20250609/depth_point_clouds"
depth_npy_files = [file for file in os.listdir(depth_dir) if file.endswith('.npy')]
# 图像目录
image_dir = "/data1/chm/datasets/wheeltec/record_20250609/images"
# odom 文件
odom_file = "/data1/chm/datasets/wheeltec/record_20250609/odom_combined/odom_combined.csv"
#
tf_map_to_odom_file = "/data1/chm/datasets/wheeltec/record_20250609/tf_data/tf_map_to_odom_combined.csv"

output_dir = "/data1/chm/datasets/wheeltec/record_20250609/outputs_dirs"

# 执行匹配
matched_df = match_all_sensors(image_dir, depth_dir, odom_file, tf_map_to_odom_file)

matched_df = matched_df.iloc[::400]
# matched_df = matched_df.iloc[::400]
# matched_df = matched_df.iloc[::400][:5]
# matched_df = matched_df.iloc[:]
print(matched_df.head())
# 生成彩色点云
color_pcd = generate_color_pointcloud(
    matched_df,
    depth_camera_intrinsic_info,
    color_camera_intrinsic_info,
    depth_dir,
    image_dir,
    depth_2_color_Transform,  # 传递变换矩阵,
    output_dir
)
o3d.io.write_point_cloud(os.path.join(output_dir, "color_pcd.ply"), color_pcd)
