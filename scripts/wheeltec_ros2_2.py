import os
import cv2
import struct
import pandas as pd
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm
from collections import defaultdict
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation as R  # 用于四元数到旋转矩阵的转换

"""
pip install opencv-python pandas numpy open3d matplotlib pyyaml tqdm rosbags scipy
"""


def load_metadata(bag_path):
    """加载metadata.yaml文件"""
    metadata_path = os.path.join(bag_path, 'metadata.yaml')
    with open(metadata_path, 'r') as f:
        metadata = yaml.safe_load(f)
    return metadata


def get_total_messages(metadata):
    """从metadata中获取总消息数"""
    return metadata['rosbag2_bagfile_information']['message_count']


def save_point_cloud_as_ply(data, file_path):
    """将点云数据保存为 .ply 文件，颜色使用 Spectral_r 颜色映射"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:, :3])  # XYZ 坐标

    if data.shape[1] == 3:
        # 使用 z 轴高度进行归一化着色
        z = data[:, 2]
        z_min, z_max = np.min(z), np.max(z)
        z_norm = (z - z_min) / (z_max - z_min + 1e-8)
        colors = plt.get_cmap('Spectral_r')(z_norm)[:, :3]  # 只取 RGB，不要 alpha 通道
        pcd.colors = o3d.utility.Vector3dVector(colors)
    elif data.shape[1] == 4:
        # 使用强度值进行归一化着色
        intensity = data[:, 3]
        intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)
        colors = plt.get_cmap('Spectral_r')(intensity_norm)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(file_path, pcd)
    # print(f"Saved point cloud to {file_path}")


def read_point_cloud2_message(msg):
    """从 sensor_msgs/PointCloud2 消息中读取点云数据"""
    fields = {f.name: f for f in msg.fields}
    field_names = [f.name for f in msg.fields]
    field_offsets = [f.offset for f in msg.fields]

    point_step = msg.point_step
    data = msg.data

    points = []
    for i in range(0, len(data), point_step):
        point = data[i:i + point_step]
        x = struct.unpack_from('f', point, fields['x'].offset)[0]
        y = struct.unpack_from('f', point, fields['y'].offset)[0]
        z = struct.unpack_from('f', point, fields['z'].offset)[0]

        if "intensity" in fields:
            intensity = struct.unpack_from('f', point, fields['intensity'].offset)[0]
            points.append([x, y, z, intensity])
        else:
            points.append([x, y, z])

    return np.array(points, dtype=np.float32)


def read_depth_point_cloud_message(msg):
    """从深度相机点云消息中读取数据"""
    fields = {f.name: f for f in msg.fields}

    # 确保字段存在且类型为 float32（datatype=7）
    for name in ['x', 'y', 'z']:
        if name not in fields:
            raise ValueError(f"Missing field: {name}")
        if fields[name].datatype != 7:
            raise ValueError(f"Unsupported datatype in field {name} (expected FLOAT32)")

    # 提取字段偏移
    offsets = {
        'x': fields['x'].offset,
        'y': fields['y'].offset,
        'z': fields['z'].offset
    }

    # 每个点的字节长度
    point_step = msg.point_step
    data = msg.data
    num_points = len(data) // point_step

    # 初始化点云数组
    points = []

    for i in range(num_points):
        base = i * point_step
        x = struct.unpack_from('f', data, base + offsets['x'])[0]
        y = struct.unpack_from('f', data, base + offsets['y'])[0]
        z = struct.unpack_from('f', data, base + offsets['z'])[0]
        if np.isnan(x) or np.isnan(y) or np.isnan(z):
            continue
        points.append([z, -x, -y])  # 激光点云的xyz分别指向(右,下,前), 重置正方向分别为(前,左,上)

    return np.array(points)


def save_imu_data(imu_data_list, output_file):
    """将 IMU 数据保存为 CSV 文件"""
    df = pd.DataFrame(imu_data_list)
    df.to_csv(output_file, index=False)
    print(f"Saved IMU data to {output_file}")


def save_odom_data(odom_data_list, output_file):
    """将里程计数据保存为 CSV 文件"""
    df = pd.DataFrame(odom_data_list)
    df.to_csv(output_file, index=False)
    print(f"Saved odom data to {output_file}")


def save_pose_data(pose_data_list, output_file):
    """将位姿数据保存为 CSV 文件"""
    df = pd.DataFrame(pose_data_list)
    df.to_csv(output_file, index=False)
    print(f"Saved pose data to {output_file}")


def save_camera_info(camera_info_list, output_file):
    """将相机信息保存为 CSV 文件"""
    df = pd.DataFrame(camera_info_list)
    df.to_csv(output_file, index=False)
    print(f"Saved camera info to {output_file}")


def quaternion_to_rotation_matrix(quaternion):
    """将四元数转换为旋转矩阵"""
    r = R.from_quat([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
    return r.as_matrix()


def pose_to_homogeneous_matrix(pose):
    """将位姿转换为齐次变换矩阵"""
    position = np.array([pose.position.x, pose.position.y, pose.position.z])
    rotation_matrix = quaternion_to_rotation_matrix(pose.orientation)
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotation_matrix
    homogeneous_matrix[:3, 3] = position
    return homogeneous_matrix


def process_tf_message(msg, timestamp_str, is_static=False):
    """处理TF消息，返回转换信息列表"""
    transforms = []
    for transform in msg.transforms:
        translation = transform.transform.translation
        rotation = transform.transform.rotation

        # 将四元数转换为欧拉角（弧度）
        euler = R.from_quat([rotation.x, rotation.y, rotation.z, rotation.w]).as_euler('xyz')

        transforms.append({
            'timestamp': timestamp_str,
            'parent_frame': transform.header.frame_id,
            'child_frame': transform.child_frame_id,
            'translation_x': translation.x,
            'translation_y': translation.y,
            'translation_z': translation.z,
            'rotation_x': rotation.x,
            'rotation_y': rotation.y,
            'rotation_z': rotation.z,
            'rotation_w': rotation.w,
            'euler_x': euler[0],
            'euler_y': euler[1],
            'euler_z': euler[2],
            'is_static': is_static
        })
    return transforms


def save_tf_data(tf_data, output_dir):
    """保存TF数据到CSV文件"""
    if not tf_data:
        print("No TF data to save")
        return

    # 按父子坐标系分组保存
    frame_pairs = defaultdict(list)
    for transform in tf_data:
        key = f"{transform['parent_frame']}_to_{transform['child_frame']}"
        frame_pairs[key].append(transform)

    # 为每个坐标系对创建单独的文件
    for pair_name, transforms in frame_pairs.items():
        # 替换文件名中的特殊字符
        safe_pair_name = pair_name.replace('/', '_').replace(':', '_')
        output_file = os.path.join(output_dir, f"tf_{safe_pair_name}.csv")

        df = pd.DataFrame(transforms)
        # 按时间戳排序
        df = df.sort_values(by='timestamp')
        df.to_csv(output_file, index=False)
        print(f"Saved TF data to {output_file}")


# def global_pose_callback(msg):
#     x = msg.pose.position.x
#     y = msg.pose.position.y
#     z = msg.pose.position.z
#     qx, qy, qz, qw = msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w
#     print("Global Pose:", (x, y, z, qx, qy, qz, qw))

# 创建保存文件夹
bag_path = "D:/03_DataSets/record_20250609"  # ROS2 bag包目录
metadata = load_metadata(bag_path)
exp = bag_path
os.makedirs(f"{exp}/images", exist_ok=True)
os.makedirs(f"{exp}/lidar_point_clouds", exist_ok=True)
os.makedirs(f"{exp}/depth_point_clouds", exist_ok=True)
os.makedirs(f"{exp}/imu_data", exist_ok=True)
os.makedirs(f"{exp}/odom_data", exist_ok=True)
os.makedirs(f"{exp}/pose_geo", exist_ok=True)  # 新增保存位姿数据的文件夹
os.makedirs(f"{exp}/camera_info", exist_ok=True)  # 新增保存相机信息的文件夹

os.makedirs(f"{exp}/global_pose", exist_ok=True)
os.makedirs(f"{exp}/localization_pose", exist_ok=True)
os.makedirs(f"{exp}/odom_combined", exist_ok=True)
os.makedirs(f"{exp}/map_data", exist_ok=True)  # 保存map和mapData
os.makedirs(f"{exp}/scan_data", exist_ok=True)  # 用于保存激光雷达扫描数据

# 初始化数据列表
imu_data_list = []
odom_data_list = []
pose_data_list = []
color_camera_info_list = []
depth_camera_info_list = []
tf_data_list = []  # 新增TF数据列表
tf_static_data_list = []  # 新增静态TF数据列表

global_pose_list = []
localization_pose_list = []
odom_combined_list = []
map_data_list = []  # 用于保存map和mapData
scan_data_list = []
imu_raw_data_list = []
# num_image = 0
# print(bag_path)
# bag_path=os.path.join(bag_path,'record_20250607_18_0.db3')
with Reader(bag_path) as reader:
    total_msgs = get_total_messages(metadata)

    # 遍历所有连接和消息
    with tqdm(total=total_msgs, desc="Processing bag file", unit='msgs') as pbar:
        for connection, timestamp, rawdata in reader.messages():
            try:
                # 反序列化消息
                typestore = get_typestore(
                    Stores.ROS2_HUMBLE)  # 实际的ROS2版本: [ROS2_FOXY, ROS2_GALACTIC, ROS2_HUMBLE, ROS2_ROLLING]
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

                # 将时间戳转换为日期格式
                timestamp_sec = timestamp / 1e9  # 转换为秒
                timestamp_str = str(timestamp_sec)

                if connection.topic == "/camera/color/image_raw":
                    # 处理彩色图像数据
                    image_path = os.path.join(f"{exp}/images", f"{timestamp_str}.png")
                    if os.path.exists(image_path):
                        continue
                    image_data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                    image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)  # RGB 转 BGR

                    cv2.imwrite(image_path, image_data)
                    # print(f"Saved color image to {image_path}")

                elif connection.topic == "/camera/depth/image_raw":
                    depth_path = os.path.join(f"{exp}/depth_point_clouds", f"{timestamp_str}.npy")
                    if os.path.exists(depth_path):
                        continue
                    # 处理深度图像数据
                    depth_data = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
                    np.save(depth_path, depth_data)
                    # print(f"Saved depth image to {depth_path}")

                elif connection.topic == "/point_cloud_raw":
                    point_cloud_path = os.path.join(f"{exp}/lidar_point_clouds", f"{timestamp_str}.ply")
                    if os.path.exists(point_cloud_path):
                        continue
                    # 处理激光雷达点云数据
                    point_cloud_data = read_point_cloud2_message(msg)
                    save_point_cloud_as_ply(point_cloud_data, point_cloud_path)

                elif connection.topic == "/camera/depth/points":
                    point_cloud_path = os.path.join(f"{exp}/depth_point_clouds", f"{timestamp_str}_depth.ply")
                    if os.path.exists(point_cloud_path):
                        continue
                    # 处理深度相机点云数据
                    # point_cloud_data = read_depth_point_cloud_message(msg)
                    # save_point_cloud_as_ply(point_cloud_data, point_cloud_path)

                elif connection.topic == "/imu/data":
                    # 处理IMU数据
                    imu_data = {
                        "timestamp": timestamp_str,
                        "orientation_x": msg.orientation.x,
                        "orientation_y": msg.orientation.y,
                        "orientation_z": msg.orientation.z,
                        "orientation_w": msg.orientation.w,
                        "angular_velocity_x": msg.angular_velocity.x,
                        "angular_velocity_y": msg.angular_velocity.y,
                        "angular_velocity_z": msg.angular_velocity.z,
                        "linear_acceleration_x": msg.linear_acceleration.x,
                        "linear_acceleration_y": msg.linear_acceleration.y,
                        "linear_acceleration_z": msg.linear_acceleration.z,
                    }
                    imu_data_list.append(imu_data)

                elif connection.topic == "/odom":
                    # 处理里程计数据
                    homogeneous_matrix = pose_to_homogeneous_matrix(msg.pose.pose)
                    odom_data = {
                        "timestamp": timestamp_str,
                        "position_x": msg.pose.pose.position.x,
                        "position_y": msg.pose.pose.position.y,
                        "position_z": msg.pose.pose.position.z,
                        "orientation_x": msg.pose.pose.orientation.x,
                        "orientation_y": msg.pose.pose.orientation.y,
                        "orientation_z": msg.pose.pose.orientation.z,
                        "orientation_w": msg.pose.pose.orientation.w,
                        "linear_velocity_x": msg.twist.twist.linear.x,
                        "linear_velocity_y": msg.twist.twist.linear.y,
                        "linear_velocity_z": msg.twist.twist.linear.z,
                        "angular_velocity_x": msg.twist.twist.angular.x,
                        "angular_velocity_y": msg.twist.twist.angular.y,
                        "angular_velocity_z": msg.twist.twist.angular.z,
                        "homogeneous_matrix": homogeneous_matrix.flatten().tolist()
                    }
                    odom_data_list.append(odom_data)

                elif connection.topic == "/pose_geo":
                    # 处理位姿数据
                    pose_data = {
                        "timestamp": timestamp_str,
                        "position_x": msg.pose.position.x,
                        "position_y": msg.pose.position.y,
                        "position_z": msg.pose.position.z,
                        "orientation_x": msg.pose.orientation.x,
                        "orientation_y": msg.pose.orientation.y,
                        "orientation_z": msg.pose.orientation.z,
                        "orientation_w": msg.pose.orientation.w
                    }
                    pose_data_list.append(pose_data)

                elif connection.topic == "/camera/color/camera_info":
                    # 处理彩色相机标定信息
                    camera_info = {
                        "timestamp": timestamp_str,
                        "height": msg.height,
                        "width": msg.width,
                        "distortion_model": msg.distortion_model,
                        "D": msg.d.tolist(),
                        "K": msg.k.tolist(),
                        "R": msg.r.tolist(),
                        "P": msg.p.tolist(),
                        "binning_x": msg.binning_x,
                        "binning_y": msg.binning_y,
                        "roi_x_offset": msg.roi.x_offset,
                        "roi_y_offset": msg.roi.y_offset,
                        "roi_height": msg.roi.height,
                        "roi_width": msg.roi.width,
                        "roi_do_rectify": msg.roi.do_rectify
                    }
                    color_camera_info_list.append(camera_info)

                elif connection.topic == "/camera/depth/camera_info":
                    # 处理深度相机标定信息
                    camera_info = {
                        "timestamp": timestamp_str,
                        "height": msg.height,
                        "width": msg.width,
                        "distortion_model": msg.distortion_model,
                        "D": msg.d.tolist(),
                        "K": msg.k.tolist(),
                        "R": msg.r.tolist(),
                        "P": msg.p.tolist(),
                        "binning_x": msg.binning_x,
                        "binning_y": msg.binning_y,
                        "roi_x_offset": msg.roi.x_offset,
                        "roi_y_offset": msg.roi.y_offset,
                        "roi_height": msg.roi.height,
                        "roi_width": msg.roi.width,
                        "roi_do_rectify": msg.roi.do_rectify
                    }
                    depth_camera_info_list.append(camera_info)

                elif connection.topic == "/tf":
                    # 处理动态TF数据
                    transforms = process_tf_message(msg, timestamp_str, is_static=False)
                    tf_data_list.extend(transforms)

                elif connection.topic == "/tf_static":
                    # 处理静态TF数据
                    transforms = process_tf_message(msg, timestamp_str, is_static=True)
                    tf_static_data_list.extend(transforms)

                elif connection.topic == "/camera/ir/image_raw":
                    continue

                elif connection.topic == "/camera/ir/camera_info":
                    continue

                elif connection.topic in ["/global_pose", "/localization_pose"]:
                    pose_data = {
                        "timestamp": timestamp_str,
                        "position_x": msg.pose.position.x,
                        "position_y": msg.pose.position.y,
                        "position_z": msg.pose.position.z,
                        "orientation_x": msg.pose.orientation.x,
                        "orientation_y": msg.pose.orientation.y,
                        "orientation_z": msg.pose.orientation.z,
                        "orientation_w": msg.pose.orientation.w
                    }
                    if connection.topic == "/global_pose":
                        global_pose_list.append(pose_data)
                    else:
                        localization_pose_list.append(pose_data)
                elif connection.topic == "/odom_combined":
                    homogeneous_matrix = pose_to_homogeneous_matrix(msg.pose.pose)
                    odom_data = {
                        "timestamp": timestamp_str,
                        "position_x": msg.pose.pose.position.x,
                        "position_y": msg.pose.pose.position.y,
                        "position_z": msg.pose.pose.position.z,
                        "orientation_x": msg.pose.pose.orientation.x,
                        "orientation_y": msg.pose.pose.orientation.y,
                        "orientation_z": msg.pose.pose.orientation.z,
                        "orientation_w": msg.pose.pose.orientation.w,
                        "linear_velocity_x": msg.twist.twist.linear.x,
                        "linear_velocity_y": msg.twist.twist.linear.y,
                        "linear_velocity_z": msg.twist.twist.linear.z,
                        "angular_velocity_x": msg.twist.twist.angular.x,
                        "angular_velocity_y": msg.twist.twist.angular.y,
                        "angular_velocity_z": msg.twist.twist.angular.z,
                        "homogeneous_matrix": homogeneous_matrix.flatten().tolist()
                    }
                    odom_combined_list.append(odom_data)

                elif connection.topic in ["/map"]:
                    map_info = {
                        "timestamp": timestamp_str,
                        "resolution": msg.info.resolution,
                        "width": msg.info.width,
                        "height": msg.info.height,
                        "origin_x": msg.info.origin.position.x,
                        "origin_y": msg.info.origin.position.y,
                        "origin_z": msg.info.origin.position.z,
                        "data": list(msg.data)  # 保存地图数据（需注意可能的数据量）
                    }
                    map_data_list.append(map_info)
                elif connection.topic == '/mapData':
                    map_data = {
                        11
                    }
                    print(map_data)

                elif connection.topic == "/scan_raw" or connection.topic == "/scan":
                    # 处理激光雷达扫描数据 (sensor_msgs/LaserScan)
                    scan_data = {
                        "timestamp": timestamp_str,
                        "angle_min": msg.angle_min,
                        "angle_max": msg.angle_max,
                        "angle_increment": msg.angle_increment,
                        "time_increment": msg.time_increment,
                        "scan_time": msg.scan_time,
                        "range_min": msg.range_min,
                        "range_max": msg.range_max,
                        "ranges": list(msg.ranges),  # 转换为列表方便保存
                        "intensities": list(msg.intensities) if msg.intensities else []
                    }
                    scan_data_list.append(scan_data)  # 需提前初始化 scan_data_list = []

                elif connection.topic == "/imu/data_raw":
                    # 处理原始IMU数据 (sensor_msgs/Imu)
                    imu_raw_data = {
                        "timestamp": timestamp_str,
                        "orientation_x": msg.orientation.x,
                        "orientation_y": msg.orientation.y,
                        "orientation_z": msg.orientation.z,
                        "orientation_w": msg.orientation.w,
                        "angular_velocity_x": msg.angular_velocity.x,
                        "angular_velocity_y": msg.angular_velocity.y,
                        "angular_velocity_z": msg.angular_velocity.z,
                        "linear_acceleration_x": msg.linear_acceleration.x,
                        "linear_acceleration_y": msg.linear_acceleration.y,
                        "linear_acceleration_z": msg.linear_acceleration.z,
                    }
                    imu_raw_data_list.append(imu_raw_data)  # 需提前初始化 imu_raw_data_list = []

                else:
                    print(f"Topic {connection.topic} not supported")
            except Exception as e:
                print(e)
            pbar.update(1)

# 保存所有数据到文件
if imu_data_list:
    imu_output_file = os.path.join(f"{exp}/imu_data", "imu_data.csv")
    save_imu_data(imu_data_list, imu_output_file)

if odom_data_list:
    odom_output_file = os.path.join(f"{exp}/odom_data", "odom_data.csv")
    save_odom_data(odom_data_list, odom_output_file)

if pose_data_list:
    pose_output_file = os.path.join(f"{exp}/pose_geo", "pose_geo.csv")
    save_pose_data(pose_data_list, pose_output_file)

if color_camera_info_list:
    camera_info_output_file = os.path.join(f"{exp}/camera_info", "color_camera_info.csv")
    save_camera_info(color_camera_info_list, camera_info_output_file)

if depth_camera_info_list:
    camera_info_output_file = os.path.join(f"{exp}/camera_info", "depth_camera_info.csv")
    save_camera_info(depth_camera_info_list, camera_info_output_file)

# 保存TF数据
if tf_data_list or tf_static_data_list:
    os.makedirs(f"{exp}/tf_data", exist_ok=True)

    # 合并静态和动态TF数据
    all_tf_data = tf_data_list + tf_static_data_list

    # 保存完整的TF数据
    save_tf_data(all_tf_data, f"{exp}/tf_data")

    # 单独保存静态TF数据（通常只需要保存一次）
    if tf_static_data_list:
        static_output_file = os.path.join(f"{exp}/tf_data", "tf_static_all.csv")
        df_static = pd.DataFrame(tf_static_data_list)
        df_static.to_csv(static_output_file, index=False)
        print(f"Saved static TF data to {static_output_file}")

if global_pose_list:
    save_pose_data(global_pose_list, os.path.join(f"{exp}/global_pose", "global_pose.csv"))
if localization_pose_list:
    save_pose_data(localization_pose_list, os.path.join(f"{exp}/localization_pose", "localization_pose.csv"))
if odom_combined_list:
    save_odom_data(odom_combined_list, os.path.join(f"{exp}/odom_combined", "odom_combined.csv"))
if map_data_list:
    pd.DataFrame(map_data_list).to_csv(os.path.join(f"{exp}/map_data", "map_data.csv"), index=False)

# if scan_data_list:
#     output_file = os.path.join(f"{exp}/scan_data", "scan_data.csv")
#     pd.DataFrame(scan_data_list).to_csv(output_file, index=False)
#     print(f"Saved scan data to {output_file}")

if imu_raw_data_list:
    output_file = os.path.join(f"{exp}/imu_data", "imu_data_raw.csv")
    pd.DataFrame(imu_raw_data_list).to_csv(output_file, index=False)
    print(f"Saved raw IMU data to {output_file}")

print("All data extraction completed!")
