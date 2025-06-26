import argparse
import json
import logging
import os
import shutil
import sys
from os.path import abspath, dirname, exists, join, basename

import cv2
import gin
import numpy as np
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline
from tqdm import trange
import pandas as pd

sys.path.append(abspath(join(dirname(__file__), '..')))
from utils_3d import fuse_mesh


def get_closest_timestamp(reference_timestamps: np.ndarray,
                          target_timestamps: np.ndarray):
    """
    This function returns:
      min_time_delta: for each time in reference_timetamps, the minimum time difference (dt) w.r.t target_timestamps
      target_index: the index of element in target_timestamps that gives minimum dt
      minimum_margin: the time difference of minimum timestamps and second minimum, used for checking uniqueness of minima
    """
    time_delta = np.abs(
        reference_timestamps.reshape(-1, 1) - target_timestamps.reshape(1, -1))

    min_two_idx = time_delta.argsort(axis=1)[:, :2]
    target_index = min_two_idx[:, 0]
    min_time_delta = time_delta[np.arange(target_index.shape[0]), target_index]
    minimum_margin = time_delta[np.arange(target_index.shape[0]),
    min_two_idx[:, 1]] - min_time_delta

    return min_time_delta, target_index, minimum_margin


def load_intrinsics(file):
    # as define here https://github.com/apple/ARKitScenes/blob/951af73d20406acf608061c16774f770c61b1405/threedod/benchmark_scripts/utils/tenFpsDataLoader.py#L46
    w, h, fx, fy, hw, hh = np.loadtxt(file)
    return np.asarray([[fx, 0, hw], [0, fy, hh], [0, 0, 1]])


def rotate_image(img, direction):
    if direction == 'Up':
        pass
    elif direction == 'Left':
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif direction == 'Right':
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif direction == 'Down':
        img = cv2.rotate(img, cv2.ROTATE_180)
    else:
        raise Exception(f'No such direction (={direction}) rotation')
    return img


def adjust_pose(pose_matrix, rotation_direction):
    """根据旋转方向调整位姿矩阵"""
    if rotation_direction == 'Left':  # 顺时针旋转图像 -> 相机坐标系逆时针转
        rot = np.array([[0, 1, 0],
                        [-1, 0, 0],
                        [0, 0, 1]])
    elif rotation_direction == 'Right':  # 逆时针旋转图像
        rot = np.array([[0, -1, 0],
                        [1, 0, 0],
                        [0, 0, 1]])
    elif rotation_direction == 'Down':
        rot = np.array([[-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, 1]])
    else:  # 'Up'
        return pose_matrix.copy()

    # 修正: 应用旋转到位姿的旋转部分，必须是后乘（右乘）
    # 这会旋转相机自身的局部坐标系
    adjusted_pose = pose_matrix.copy()
    adjusted_pose[:3, :3] = adjusted_pose[:3, :3] @ rot
    return adjusted_pose


def adjust_intrinsic(intrinsic_matrix, rotation_direction, H, W):
    """根据旋转方向调整内参矩阵"""
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    if rotation_direction == 'Left':  # 顺时针90度
        # 修正: 新主点 (c'_x, c'_y) 应该是 (H - cy, cx)
        new_intrinsic = np.array([
            [fy, 0, H - cy],
            [0, fx, cx],
            [0, 0, 1]
        ])
    elif rotation_direction == 'Right':  # 逆时针90度
        # 修正: 新主点 (c'_x, c'_y) 应该是 (cy, W - cx)
        new_intrinsic = np.array([
            [fy, 0, cy],
            [0, fx, W - cx],
            [0, 0, 1]
        ])
    elif rotation_direction == 'Down':  # 180度
        new_intrinsic = np.array([
            [fx, 0, W - cx],
            [0, fy, H - cy],
            [0, 0, 1]
        ])
    else:  # 'Up'或不旋转
        return intrinsic_matrix.copy()

    return new_intrinsic


@gin.configurable
def process_arkit(
        scan_dir: str,
        target_dir: str,
        sdf_trunc: float,
        voxel_length: float,
        depth_trunc: float,
):
    logger = logging.getLogger('ARKitProcess')
    logger.setLevel(logging.DEBUG)
    consoleHeader = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    consoleHeader.setFormatter(formatter)
    logger.addHandler(consoleHeader)

    logger.info(
        "Processing ARKitScene scan to LabelMaker format, from {} to {}...".
        format(scan_dir, target_dir))

    color_dir = join(scan_dir, 'vga_wide')
    intrinsic_dir = join(scan_dir, 'vga_wide_intrinsics')

    depth_dir = join(scan_dir, 'lowres_depth')
    confidence_dir = join(scan_dir, 'confidence')

    trajectory_file = join(scan_dir, 'lowres_wide.traj')

    meta_data_csv_file = join(os.path.join(scan_dir, "../.."), 'metadata.csv')
    assert exists(color_dir), "vga_wide attribute not downloaded!"
    assert exists(intrinsic_dir), "vga_wide_intrinsics attribute not downloaded!"
    assert exists(depth_dir), "lowres_depth attribute not downloaded!"
    assert exists(confidence_dir), "confidence attribute not downloaded!"
    assert exists(trajectory_file), "lowres_wide.traj attribute not downloaded!"
    assert exists(meta_data_csv_file), "metadata.csv  not downloaded!"

    video_id = int(basename(scan_dir))
    meta_data = pd.read_csv(meta_data_csv_file)
    sky_direction = meta_data.loc[meta_data['video_id'] == video_id, 'sky_direction'].values[0]
    if sky_direction == 'Down':
        sky_direction = 'Up'

    color_file_list = os.listdir(color_dir)
    depth_file_list = os.listdir(depth_dir)
    confidence_file_list = os.listdir(confidence_dir)
    intr_file_list = os.listdir(intrinsic_dir)

    # ts stands for timestamps, inv stands for inverse
    color_ts, color_inv = np.unique(
        np.array([
            float(name.split('_')[1].split('.png')[0]) for name in color_file_list
        ]),
        return_index=True,
    )
    depth_ts, depth_inv = np.unique(
        np.array([
            float(name.split('_')[1].split('.png')[0]) for name in depth_file_list
        ]),
        return_index=True,
    )
    confidence_ts, confidence_inv = np.unique(
        np.array([
            float(name.split('_')[1].split('.png')[0])
            for name in confidence_file_list
        ]),
        return_index=True,
    )
    intrinsic_ts, intrinsic_inv = np.unique(
        np.array([
            float(name.split('_')[1].split('.pincam')[0])
            for name in intr_file_list
        ]),
        return_index=True,
    )

    # load trajactory
    trajectory_data = np.loadtxt(trajectory_file, delimiter=' ')
    trajectory_ts = trajectory_data[:, 0]  # already sorted

    # synchronization
    logger.info("Synchronizing timestamps...")
    dt_max = 1 / 60 / 2  # half of frame time step

    # we compare all with respect to color, as color folder is sparser
    # if the matched timestamp and second matched timestamp have difference less than 1 milisecond,
    # we regard this case as the matching is not unique, and throw a warning.
    margin_threshold = 1e-3
    depth_dt, depth_idx, depth_margin = get_closest_timestamp(color_ts, depth_ts)
    if depth_margin.min() < margin_threshold:
        logger.warn(
            "Found multiple color timestamps matching in timestamps: {}".format(
                color_ts[depth_margin < margin_threshold].tolist()))

    confidence_dt, confidence_idx, confidence_margin = get_closest_timestamp(color_ts, confidence_ts)
    if confidence_margin.min() < margin_threshold:
        logger.warn(
            "Found multiple confidence timestamps matching in timestamps: {}".
            format(color_ts[confidence_margin < margin_threshold].tolist()))

    intrinsic_dt, intrinsic_idx, intrinsic_margin = get_closest_timestamp(color_ts, intrinsic_ts)
    if intrinsic_margin.min() < margin_threshold:
        logger.warn(
            "Found multiple intrinsic timestamps matching in timestamps: {}".format(
                color_ts[intrinsic_margin < margin_threshold].tolist()))

    color_idx = np.arange(color_ts.shape[0])

    # we also want to interpolate pose, so we have to filter out times outside trajectory timestamp
    timestamp_filter = (depth_dt < dt_max) * (confidence_dt < dt_max) * (
            intrinsic_dt < dt_max) * (color_ts >= trajectory_ts.min()) * (
                               color_ts <= trajectory_ts.max())

    timestamp = color_ts[timestamp_filter]
    logger.info("Synchronization finished!")

    if depth_dt[timestamp_filter].max(
    ) > 1e-8 or confidence_dt[timestamp_filter].max(
    ) > 1e-8 or intrinsic_dt[timestamp_filter].max() > 1e-8:
        depth_unmatched = depth_dt[timestamp_filter].max() > 1e-8
        intrinsic_unmatched = intrinsic_dt[timestamp_filter].max() > 1e-8
        confidence_unmatched = confidence_dt[timestamp_filter].max() > 1e-8

        unmatched_timestamp = timestamp[depth_unmatched + intrinsic_unmatched +
                                        confidence_unmatched].tolist()
        logger.info("There are not perfectly matched timestamps: {}".format(
            unmatched_timestamp))

    # interpolate pose
    logger.info("Interpolating poses...")
    rots = Rotation.from_rotvec(trajectory_data[:, 1:4])
    rot_spline = RotationSpline(trajectory_ts, rots)

    x_spline = CubicSpline(trajectory_ts, trajectory_data[:, 4])
    y_spline = CubicSpline(trajectory_ts, trajectory_data[:, 5])
    z_spline = CubicSpline(trajectory_ts, trajectory_data[:, 6])

    num_frame = timestamp_filter.sum()

    extrinsics_mat = np.zeros(shape=(num_frame, 4, 4))
    extrinsics_mat[:, 3, 3] = 1.0
    extrinsics_mat[:, :3, :3] = rot_spline(timestamp).as_matrix()
    extrinsics_mat[:, :3, 3] = np.stack(
        [x_spline(timestamp),
         y_spline(timestamp),
         z_spline(timestamp)], axis=1)
    pose_mat = np.linalg.inv(extrinsics_mat)
    logger.info("Pose interpolation finished!")

    # get correspondence to original file
    rows = []
    # num_frame = 100  # for debug
    for i in range(num_frame):
        frame_id = '{:06d}'.format(i)
        color_pth = color_file_list[color_inv[color_idx[timestamp_filter][i]]]
        depth_pth = depth_file_list[depth_inv[depth_idx[timestamp_filter][i]]]
        confdc_pth = confidence_file_list[confidence_inv[
            confidence_idx[timestamp_filter][i]]]
        intr_pth = intr_file_list[intrinsic_inv[intrinsic_idx[timestamp_filter][i]]]
        rows.append([frame_id, color_pth, depth_pth, confdc_pth, intr_pth])

    # write to new file
    shutil.rmtree(target_dir, ignore_errors=True)
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(join(target_dir, 'color'), exist_ok=True)
    os.makedirs(join(target_dir, 'depth'), exist_ok=True)
    os.makedirs(join(target_dir, 'intrinsic'), exist_ok=True)
    os.makedirs(join(target_dir, 'pose'), exist_ok=True)

    # first write correspondence list
    fields = [
        'frame_id', 'original_color_path', 'original_depth_path',
        'original_confidence_path', 'original_intrinsic_path'
    ]
    correspondence_list = [dict(zip(fields, row)) for row in rows]
    json_object = json.dumps(correspondence_list, indent=4)
    with open(join(target_dir, 'correspondence.json'), 'w') as jsonfile:
        jsonfile.write(json_object)
    logger.info("Saved old and new files correspondence to {}.".format(
        join(target_dir, 'correspondence.json')))

    logger.info("Transfering files...")
    print("Sky Direction", sky_direction)
    for idx in trange(num_frame):
        frame_id, color_pth, depth_pth, confdc_pth, intr_pth = rows[idx]
        # save color
        tgt_color_pth = join(target_dir, 'color',
                             frame_id + '.jpg')  # png -> jpg, compressed
        color_img = Image.open(join(color_dir, color_pth))

        o_h, o_w, _ = np.asarray(color_img).shape
        color_img = rotate_image(np.asarray(color_img), sky_direction)
        color_img = Image.fromarray(color_img)

        color_img.save(tgt_color_pth)
        h, w, _ = np.asarray(color_img).shape

        # save pose
        tgt_pose_pth = join(target_dir, 'pose', frame_id + '.txt')
        pose = pose_mat[idx]
        pose = adjust_pose(pose, sky_direction)
        np.savetxt(tgt_pose_pth, pose)  #

        # process and save intr
        tgt_intrinsic_pth = join(target_dir, 'intrinsic', frame_id + '.txt')
        intrinsic = load_intrinsics(join(intrinsic_dir, intr_pth))
        intrinsic = adjust_intrinsic(intrinsic, sky_direction, o_h, o_w)
        np.savetxt(tgt_intrinsic_pth, intrinsic)

        # process and save depth
        depth = cv2.imread(join(depth_dir, depth_pth), cv2.IMREAD_UNCHANGED)
        confdc = cv2.imread(join(confidence_dir, confdc_pth), cv2.IMREAD_UNCHANGED)
        depth[confdc < 2] = 0
        depth = rotate_image(depth, sky_direction)
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

        tgt_depth_pth = join(target_dir, 'depth', frame_id + '.png')
        cv2.imwrite(tgt_depth_pth, depth)
        ##### debug
        # if idx == num_frame - 1:
        #     pcd = reproject_to_3d(tgt_color_pth, tgt_depth_pth, intrinsic, pose, depth_scale=1000)
        #     o3d.io.write_point_cloud(join(target_dir, "output.ply"), pcd)

    logger.info("File transfer finished!")

    logger.info("Fusing RGBD images into TSDF Volmue...")
    fuse_mesh(
        scan_dir=target_dir,
        sdf_trunc=sdf_trunc,
        voxel_length=voxel_length,
        depth_trunc=depth_trunc,
        depth_scale=1000.0,
    )  # depth_scale is a fixed value in ARKitScene, no need to pass an argument in cli
    logger.info("Fusion finished! Saving to file as {}".format(
        join(target_dir, 'mesh.ply')))

    ################################################################################
    frame_interval = 15  # 每15帧选1帧
    logger.info(f"Selecting 1 frame every {frame_interval} frames...")

    # 获取所有帧的文件列表
    color_files = sorted(os.listdir(join(target_dir, 'color')))

    # 筛选要保留的帧索引
    keep_indices = range(0, len(color_files), frame_interval)

    # 删除未被选中的帧
    for i in range(len(color_files)):
        if i not in keep_indices:
            frame_id = f"{i:06d}"
            os.remove(join(target_dir, 'color', f"{frame_id}.jpg"))
            os.remove(join(target_dir, 'depth', f"{frame_id}.png"))
            os.remove(join(target_dir, 'pose', f"{frame_id}.txt"))
            os.remove(join(target_dir, 'intrinsic', f"{frame_id}.txt"))
    logger.info("Frame sampling finished")


def reproject_to_3d(rgb_path, depth_path, intrinsics, pose_mat, depth_scale=1.0, depth_trunc=3.0):
    import open3d as o3d
    """
    将RGB图像反投影到3D空间

    参数:
        rgb_path: RGB图像路径
        depth_path: 深度图像路径
        intrinsics: 相机内参矩阵 (3x3)
        depth_scale: 深度比例因子 (深度值 = 像素值 / depth_scale)
        depth_trunc: 最大有效深度值 (米)

    返回:
        Open3D点云对象
    """
    # 加载图像
    color = np.array(Image.open(rgb_path).convert('RGB'))
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    # 处理深度数据
    if depth_scale > 0:
        depth = depth / depth_scale  # 转换为米单位

    # 有效深度范围截断
    depth[depth > depth_trunc] = 0

    # 获取图像尺寸
    height, width = depth.shape

    # 创建像素坐标网格
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # 归一化相机坐标
    x = (u - intrinsics[0, 2]) * depth / intrinsics[0, 0]
    y = (v - intrinsics[1, 2]) * depth / intrinsics[1, 1]
    z = depth

    # 过滤无效点
    valid = z > 0
    colors = color[valid]

    # 创建齐次坐标点
    points_homo = np.vstack((
        x[valid].ravel(),
        y[valid].ravel(),
        z[valid].ravel(),
        np.ones_like(x[valid].ravel())
    ))

    # 应用相机位姿变换
    world_points = (pose_mat @ points_homo)[:-1]  # 移除齐次坐标维度

    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(world_points.T)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    return pcd


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_dir", type=str)
    parser.add_argument("--target_dir", type=str)
    parser.add_argument("--sdf_trunc", type=float, default=0.04)
    parser.add_argument("--voxel_length", type=float, default=0.008)
    parser.add_argument("--depth_trunc", type=float, default=3.0)
    parser.add_argument('--config', help='Name of config file')

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    if args.config is not None:
        gin.parse_config_file(args.config)
    process_arkit(
        scan_dir=args.scan_dir,
        target_dir=args.target_dir,
        sdf_trunc=args.sdf_trunc,
        voxel_length=args.voxel_length,
        depth_trunc=args.depth_trunc,
    )
