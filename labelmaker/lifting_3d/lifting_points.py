import argparse
import logging
import os
from pathlib import Path
from typing import Union

import cv2
import gin
import numpy as np
import open3d as o3d
from PIL import Image
from tqdm import tqdm

from labelmaker.label_data import get_wordnet

logging.basicConfig(level="INFO")
log = logging.getLogger('3D Point Lifting')


def project_pointcloud(points, pose, intrinsics):
    """将3D点云投影到2D图像平面

    该函数通过给定的相机位姿和相机内参，将3D点云投影到2D图像平面。
    计算过程包括：将点云转换到相机坐标系，然后应用相机内参进行投影，
    最后进行归一化处理得到2D坐标。

    Args:
        points: numpy数组，形状为(N,3)，表示N个3D点的坐标(x,y,z)
        pose: numpy数组，形状为(4,4)，表示相机在世界坐标系中的位姿(变换矩阵)
        intrinsics: numpy数组，形状为(3,3)或(3,4)，表示相机内参矩阵

    Returns:
        numpy数组，形状为(N,3)，表示投影后的2D齐次坐标(x,y,w)，
        其中w是深度值，x/w和y/w是实际的2D坐标
    """
    # 将3D点转换为齐次坐标(N,4)
    points_h = np.hstack((points, np.ones_like(points[:, 0:1])))
    # 将点从世界坐标系转换到相机坐标系
    points_c = np.linalg.inv(pose) @ points_h.T
    points_c = points_c.T
    # 如果内参矩阵是3x3，则扩展为4x4的齐次形式
    if intrinsics.shape[-1] == 3:
        intrinsics = np.hstack((intrinsics, np.zeros((3, 1))))
        intrinsics = np.vstack((intrinsics, np.zeros((1, 4))))
        intrinsics[-1, -1] = 1.
    # 应用内参矩阵进行投影变换
    points_p = intrinsics @ points_c.T
    points_p = points_p.T[:, :3]
    # 归一化处理，将齐次坐标转换为2D坐标
    points_p[:, 0] /= (points_p[:, -1] + 1.e-6)
    points_p[:, 1] /= (points_p[:, -1] + 1.e-6)

    return points_p


@gin.configurable
def main(
        scene_dir: Union[str, Path],
        label_folder: Union[str, Path],
        output_file: Union[str, Path],
        output_mesh: Union[str, Path],
        maximum_label: int,
        label_key='occ11_id',
        output_key='occ11_id',
):
    scene_dir = Path(scene_dir)
    label_folder = Path(label_folder)
    output_file = Path(output_file)
    output_mesh = Path(output_mesh)

    # check if scene_dir exists
    assert scene_dir.exists() and scene_dir.is_dir()

    # define all paths
    input_color_dir = scene_dir / 'color'
    assert input_color_dir.exists() and input_color_dir.is_dir()

    input_depth_dir = scene_dir / 'depth'
    assert input_depth_dir.exists() and input_depth_dir.is_dir()

    input_intrinsic_dir = scene_dir / 'intrinsic'
    assert input_intrinsic_dir.exists() and input_intrinsic_dir.is_dir()

    input_pose_dir = scene_dir / 'pose'
    assert input_pose_dir.exists() and input_pose_dir.is_dir()

    input_label_dir = scene_dir / label_folder
    assert input_label_dir.exists() and input_label_dir.is_dir()

    input_mesh_path = scene_dir / 'mesh.ply'
    assert input_mesh_path.exists() and input_mesh_path.is_file()

    log.info('Processing {} using for labels {}'.format(
        str(scene_dir),
        str(input_label_dir),
    ))

    # load mesh and extract colors
    mesh = o3d.io.read_triangle_mesh(str(input_mesh_path))
    vertices = np.asarray(mesh.vertices)

    # init label container
    labels_3d = np.zeros((vertices.shape[0], maximum_label + 1))

    files = input_label_dir.glob('*.png')
    files = sorted(files, key=lambda x: int(x.stem.split('.')[0]))
    resize_image = False
    # 把 3d mesh 再投影至 2D image, 再统计所有的
    for idx, file in tqdm(enumerate(files), total=len(files)):

        frame_key = file.stem

        # 加载相机内参
        intrinsics = np.loadtxt(str(input_intrinsic_dir / f'{frame_key}.txt'))

        # 加载RGB图像
        image = np.asarray(Image.open(str(input_color_dir /
                                          f'{frame_key}.jpg'))).astype(np.uint8)  #
        # 加载深度图
        depth = np.asarray(Image.open(str(
            input_depth_dir / f'{frame_key}.png'))).astype(np.float32) / 1000.

        # 加载语义图
        labels = np.asarray(Image.open(str(file)))

        #
        max_label = np.max(labels)
        if max_label > labels_3d.shape[-1] - 1:
            raise ValueError(
                f'Label {max_label} is not in the label range of {labels_3d.shape[-1]}'
            )

        # 图像缩放
        if resize_image:
            h, w = depth.shape
            image = cv2.resize(image, (w, h))
            labels = cv2.resize(labels, (w, h))
        else:
            h, w, _ = image.shape
            depth = cv2.resize(depth, (w, h))
        # 加载相机位姿
        pose_file = input_pose_dir / f'{frame_key}.txt'
        pose = np.loadtxt(str(pose_file))

        points_p = project_pointcloud(vertices, pose, intrinsics)

        xx = points_p[:, 0].astype(int)
        yy = points_p[:, 1].astype(int)
        zz = points_p[:, 2]

        valid_mask = (xx >= 0) & (yy >= 0) & (xx < w) & (yy < h)

        d = depth[yy[valid_mask], xx[valid_mask]]

        valid_mask[valid_mask] = (zz[valid_mask] > 0) & (np.abs(zz[valid_mask] - d) <= 0.1)

        labels_2d = labels[yy[valid_mask], xx[valid_mask]]
        labels_3d[valid_mask, labels_2d] += 1

    # extract labels
    labels_3d = np.argmax(labels_3d, axis=-1)

    # save output
    np.savetxt(str(scene_dir / output_file), labels_3d, fmt='%i')  # 保存 3d label

    # save colored mesh
    color_map = np.zeros(shape=(maximum_label, 3), dtype=np.uint8)
    ############# before ##########
    # for item in get_wordnet():
    #     color_map[item['id']] = item['color']
    ############# before ##########
    for item in get_wordnet(label_key='occ11_id', output_key='occ11_id'):
        color_map[item['id']] = item['color']

    label_mesh_color = color_map[labels_3d]  # 根据wordnet的映射来生成mesh color

    label_mesh = o3d.geometry.TriangleMesh()
    label_mesh.vertices = mesh.vertices
    label_mesh.triangles = mesh.triangles

    label_mesh.vertex_colors = o3d.utility.Vector3dVector(
        label_mesh_color.astype(float) / 255)

    o3d.io.write_triangle_mesh(str(scene_dir / output_mesh), label_mesh)


def arg_parser():
    parser = argparse.ArgumentParser(
        description=
        'Project 3D points to 2D image plane and aggregate labels and save label txt'
    )
    parser.add_argument(
        '--workspace',
        type=str,
        required=True,
        help=
        'Path to workspace directory. There should be a "color" folder inside.',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='labels.txt',
        help='Name of files to save the labels',
    )
    parser.add_argument(
        '--output_mesh',
        type=str,
        default='point_lifted_mesh.ply',
        help='Name of files to save the labels',
    )
    parser.add_argument('--label_folder', default='intermediate/consensus')
    parser.add_argument(
        '--max_label',
        type=int,
        default=2000,
        help='Max label value',
    )
    parser.add_argument('--config', help='Name of config file')
    parser.add_argument('--label_key', default='occ_11')
    parser.add_argument('--output_key', default='occ_11')

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parser()
    if args.config is not None:
        gin.parse_config_file(args.config)
    main(
        scene_dir=args.workspace,
        label_folder=args.label_folder,
        output_file=args.output,
        output_mesh=args.output_mesh,
        maximum_label=args.max_label,
        label_key=args.label_key,
        output_key=args.output_key
    )
