import argparse
import json
import logging
import os
import shutil
import sys
from os.path import abspath, dirname, exists, join

import cv2
import gin
import numpy as np
from PIL import Image
from tqdm import trange

sys.path.append(abspath(join(dirname(__file__), '..')))
from utils_3d import fuse_mesh
from scripts.colmap_loader import Image as colmap_Image
from scripts.colmap_loader import Camera
from scripts.colmap_loader import qvec2rotmat
from scipy.spatial.transform import Rotation


def load_intrinsics(file):
    # as define here https://github.com/apple/ARKitScenes/blob/951af73d20406acf608061c16774f770c61b1405/threedod/benchmark_scripts/utils/tenFpsDataLoader.py#L46
    w, h, fx, fy, hw, hh = np.loadtxt(file)
    return np.asarray([[fx, 0, hw], [0, fy, hh], [0, 0, 1]])


def read_extrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = colmap_Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


def read_intrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras


@gin.configurable
def process_habitat(
        scan_dir: str,
        target_dir: str,
        sdf_trunc: float,
        voxel_length: float,
        depth_trunc: float,
):
    logger = logging.getLogger('HabitatProcess')
    logger.setLevel(logging.DEBUG)
    consoleHeader = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    consoleHeader.setFormatter(formatter)
    logger.addHandler(consoleHeader)

    logger.info(
        "Processing Habitat scan to LabelMaker format, from {} to {}...".
        format(scan_dir, target_dir))

    # rgb图像
    color_dir = join(scan_dir, 'images')

    # 深度图像
    depth_dir = join(scan_dir, 'depth')

    # pose
    pose_file = join(scan_dir, 'sparse', '0', 'images.txt')

    # 相机内参
    intrinsic_file = join(scan_dir, 'sparse', '0', 'cameras.txt')

    assert exists(color_dir), "images attribute not downloaded!"
    assert exists(depth_dir), "depth attribute not downloaded!"
    assert exists(pose_file), "vga_wide_intrinsics attribute not downloaded!"
    assert exists(intrinsic_file), "camera_intrinsics.txt attribute not downloaded!"

    color_file_list = sorted(os.listdir(color_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    depth_file_list = sorted(os.listdir(depth_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))

    assert len(color_file_list) == len(depth_file_list)

    # ts stands for timestamps, inv stands for inverse
    pose_data = read_extrinsics_text(pose_file)

    intrinsic_data = read_intrinsics_text(intrinsic_file)

    num_frame = len(depth_file_list)

    # write to new file
    shutil.rmtree(target_dir, ignore_errors=True)
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(join(target_dir, 'color'), exist_ok=True)
    os.makedirs(join(target_dir, 'depth'), exist_ok=True)
    os.makedirs(join(target_dir, 'intrinsic'), exist_ok=True)
    os.makedirs(join(target_dir, 'pose'), exist_ok=True)

    #
    import open3d as o3d
    pcd1 = o3d.geometry.PointCloud()

    logger.info("Transfering files...")
    for idx in trange(num_frame):
        image_id = idx + 1

        new_name = str(image_id).zfill(6)

        # save color
        color_pth = color_file_list[idx]
        tgt_color_pth = join(target_dir, 'color', new_name + '.jpg')  # png -> jpg, compressed
        color_img = Image.open(join(color_dir, color_pth))
        if color_img.mode == 'RGBA':
            color_img = color_img.convert('RGB')
        color_img.save(tgt_color_pth)
        h, w, _ = np.asarray(color_img).shape

        # process and save depth
        depth_pth = depth_file_list[idx]
        depth = np.load(join(depth_dir, depth_pth))
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        depth = depth * 1000  # m ->mm
        depth = depth.astype(np.uint16)
        tgt_depth_pth = join(target_dir, 'depth', new_name + '.png')
        cv2.imwrite(tgt_depth_pth, depth)

        # save pose
        extr = pose_data[idx + 1]  # cam
        # R = qvec2rotmat(extr.qvec)
        # T = np.array(extr.tvec)
        # pose_matrix = np.eye(4)
        # pose_matrix[:3, :3] = R
        # pose_matrix[:3, 3] = T  # m -> mm

        position = extr.tvec
        quaternion = extr.qvec
        rotation = Rotation.from_quat(quaternion).as_matrix()
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rotation
        pose_matrix[:3, 3] = position
        transform_matrix = np.array([
            [1, 0, 0, 0],
            [0, - 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        #
        #        z  - forward                                        y -up
        #      /                                                   |
        #     /                                                    |
        #      -------> x - right            to                     -------> x - right
        #     |                                                    /
        #     |                                                   /
        #     | y down                                            z back
        #
        c2w = pose_matrix @ transform_matrix  # camera to world
        pose_w2c = np.linalg.inv(c2w)  # world to camera

        tgt_pose_pth = join(target_dir, 'pose', new_name + '.txt')
        np.savetxt(tgt_pose_pth, c2w)  # (4x4) world to camera

        # save intrinsic
        intrinsic = intrinsic_data[idx + 1]
        fx, fy, cx, cy = intrinsic.params
        intrinsic_matrix = np.eye(3)
        intrinsic_matrix[0, 0] = fx
        intrinsic_matrix[1, 1] = fy
        intrinsic_matrix[0, 2] = cx
        intrinsic_matrix[1, 2] = cy
        tgt_intrinsic_pth = join(target_dir, 'intrinsic', new_name + '.txt')
        np.savetxt(tgt_intrinsic_pth, intrinsic_matrix)  # (3x3)

        # debug
        if idx == 0 or idx == 5 or idx == 10:
            # 反投影点云
            height, width = depth.shape
            u, v = np.meshgrid(np.arange(width), np.arange(height))
            z = depth  # m unit
            # z = depth / 1000
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
            colors = np.array(color_img).reshape(-1, 3) / 255.0  # 归一化

            points_hom = np.hstack([points, np.ones((len(points), 1))])
            points = (c2w @ points_hom.T).T[:, :3]
            pcd1.points = o3d.utility.Vector3dVector(
                np.vstack([np.asarray(pcd1.points), points])
            )
            pcd1.colors = o3d.utility.Vector3dVector(
                np.vstack([np.asarray(pcd1.colors), colors]))

    o3d.io.write_point_cloud("pcd.ply", pcd1)

    logger.info("File transfer finished!")

    logger.info("Fusing RGBD images into TSDF Volmue...")
    fuse_mesh(
        scan_dir=target_dir,
        sdf_trunc=sdf_trunc,
        voxel_length=voxel_length,
        depth_trunc=depth_trunc,
        depth_scale=1000.,
    )
    logger.info("Fusion finished! Saving to file as {}".format(
        join(target_dir, 'mesh.ply')))


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan_dir", type=str)
    parser.add_argument("--target_dir", type=str)
    parser.add_argument("--sdf_trunc", type=float, default=0.04)  # 4cm
    parser.add_argument("--voxel_length", type=float, default=0.008)  # 8mm
    parser.add_argument("--depth_trunc", type=float, default=6.0)  #
    parser.add_argument('--config', help='Name of config file')

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    if args.config is not None:
        gin.parse_config_file(args.config)
    process_habitat(
        scan_dir=args.scan_dir,
        target_dir=args.target_dir,
        sdf_trunc=args.sdf_trunc,
        voxel_length=args.voxel_length,
        depth_trunc=args.depth_trunc,
    )
