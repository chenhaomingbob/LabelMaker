import argparse
import logging
import os
from pathlib import Path
from typing import Union
import gin
import cv2
import open3d as o3d
import numpy as np
import shutil

from labelmaker.label_data import get_wordnet, get_occ11
from tools import point_cloud_to_voxel_grid_unbounded
from tqdm import tqdm
from PIL import Image
from sklearn.neighbors import KDTree
import pickle
from copy import deepcopy

logging.basicConfig(level="INFO")
log = logging.getLogger('Mesh2Occupancy')

vis_debug = False


@gin.configurable
def main(
        scene_dir: Union[str, Path],
        input_label: Union[str, Path],
        input_mesh: Union[str, Path],
        output_global_occupancy: Union[str, Path],
        output_frame_folder: Union[str, Path],
        label_space='occ11',
        voxel_size=0.08
):
    scene_dir = Path(scene_dir)
    intput_label = Path(input_label)  #
    input_mesh = Path(input_mesh)
    output_global_occupancy = Path(output_global_occupancy)

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

    input_mesh_path = scene_dir / input_mesh
    assert input_mesh_path.exists() and input_mesh_path.is_file()

    input_label_path = scene_dir / intput_label
    assert input_label_path.exists() and input_label_path.is_file()

    output_dir = scene_dir / output_frame_folder
    shutil.rmtree(output_dir, ignore_errors=True)  # remove output_dir if it exists
    os.makedirs(str(output_dir), exist_ok=False)
    ##########

    # load mesh and extract colors
    mesh = o3d.io.read_triangle_mesh(str(input_mesh_path))
    vertices = np.asarray(mesh.vertices)  # (N,3)

    # load the semantic label for each point
    label_3d = np.loadtxt(str(input_label_path))  # (N,)

    assert len(label_3d) == len(vertices)

    # define color map for visualization
    num_classes = 2000
    color_map = np.zeros(shape=(num_classes, 3), dtype=np.uint8)

    if label_space == 'wordnet':
        num_classes = len(get_wordnet())
        for item in get_wordnet():
            color_map[item['id']] = item['color']
    elif label_space == 'occ11':
        num_classes = len(get_occ11())
        for item in get_occ11():
            color_map[item['id']] = item['color']
    else:
        raise Exception(f'Unknown label space {label_space}')
    ##############################
    # generate global occupancy
    ##############################
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)

    voxel_down_pcd, _, inverse_index_list = pcd.voxel_down_sample_and_trace(
        voxel_size=voxel_size, min_bound=pcd.get_min_bound(), max_bound=pcd.get_max_bound()
    )

    o3d.io.write_point_cloud(str(scene_dir / 'pc_voxel_down.ply'), voxel_down_pcd)

    voxel_votes = np.zeros((len(voxel_down_pcd.points), num_classes), dtype=int)
    all_indices = np.concatenate(inverse_index_list)
    all_sem_labels = np.concatenate([np.full(len(indices), i) for i, indices in enumerate(inverse_index_list)])
    sem_labels = label_3d[all_indices].astype(int)
    np.add.at(voxel_votes, (all_sem_labels, sem_labels), 1)
    voxel_sem = np.argmax(voxel_votes, axis=1)

    valid_point_mask = voxel_sem > 0
    voxel_down_pcd.points = o3d.utility.Vector3dVector(np.asarray(voxel_down_pcd.points)[valid_point_mask, :])
    voxel_down_pcd.colors = o3d.utility.Vector3dVector(color_map[voxel_sem[valid_point_mask]] / 255)

    o3d.io.write_point_cloud(str(scene_dir / 'pc_voxel_down_color.ply'), voxel_down_pcd)

    points = np.asarray(voxel_down_pcd.points)
    logging.info("Scene Range")
    log.info("Scene Range")
    log.info("X  max {} min {} range {}".format(points[:, 0].max(), points[:, 0].min(),
                                                points[:, 0].max() - points[:, 0].min()))
    log.info("Y  max {} min {} range {}".format(points[:, 1].max(), points[:, 1].min(),
                                                points[:, 1].max() - points[:, 1].min()))
    log.info("Z  max {} min {} range {}".format(points[:, 2].max(), points[:, 2].min(),
                                                points[:, 2].max() - points[:, 2].min()))
    scene_voxels = points
    scene_voxels_sem = voxel_sem[valid_point_mask]
    ##############################
    # generate frame occupancy
    ##############################
    if voxel_size == 0.08:
        voxDim = np.asarray([60, 60, 36])  # 0.08 cm
    elif voxel_size == 0.04:
        voxDim = np.asarray([120, 120, 72])  # 0.08 cm
    elif voxel_size == 0.02:
        voxDim = np.asarray([240, 240, 144])  # 0.02 cm

    assert np.all(voxDim * voxel_size == np.asarray([4.8, 4.8, 2.88]))
    # [4.8,4.8,2.88]
    voxOriginCam = np.asarray([
        [0], [0], [1.44]]
    )

    files = input_color_dir.glob('*.jpg')
    files = sorted(files, key=lambda x: int(x.stem.split('.')[0]))
    for idx, file in tqdm(enumerate(files), total=len(files)):

        # if idx == 0 or idx == 10 or idx == 199:
        #     pass
        # else:
        #     continue

        frame_key = file.stem

        # 加载RGB图像
        image = np.asarray(Image.open(str(input_color_dir / f'{frame_key}.jpg'))).astype(np.uint8)  #

        # 加载相机内参
        intrinsics = np.loadtxt(str(input_intrinsic_dir / f'{frame_key}.txt'))

        # 加载深度图
        depth = np.asarray(Image.open(str(input_depth_dir / f'{frame_key}.png'))).astype(np.float32) / 1000.

        # 缩放深度图
        h, w, _ = image.shape
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

        # 加载相机位姿
        pose_file = input_pose_dir / f'{frame_key}.txt'
        pose = np.loadtxt(str(pose_file))  # camera to world

        cam2world = pose

        voxOriginWorld = cam2world[:3, :3] @ voxOriginCam + cam2world[:3, -1:]
        voxOriginWorld2 = deepcopy(voxOriginWorld)
        delta = np.array(
            [[2.4],
             [2.4],
             [1.44]]
        )  # 世界坐标系下的
        voxOriginWorld -= delta

        # 保留距离原点4.8范围内的场景
        scene_voxels_delta = np.abs(scene_voxels[:, :3] - voxOriginWorld.reshape(-1))
        mask = np.logical_and(scene_voxels_delta[:, 0] <= 4.8,
                              np.logical_and(scene_voxels_delta[:, 1] <= 4.8,
                                             scene_voxels_delta[:, 2] <= 4.8))
        frame_voxels = scene_voxels[mask]
        frame_voxels_sem = scene_voxels_sem[mask]
        # ================> debug
        if vis_debug:
            frame_pcd1 = o3d.geometry.PointCloud()
            frame_voxels_xyz = np.vstack((frame_voxels,
                                          voxOriginWorld.reshape(-1).reshape((1, 3)),
                                          voxOriginWorld2.reshape(-1).reshape((1, 3)),

                                          ))
            frame_voxels_color = np.vstack((color_map[scene_voxels_sem[mask]],
                                            [214, 38, 40],  # 红色
                                            [43, 160, 43],  # 绿色
                                            )
                                           )
            frame_pcd1.points = o3d.utility.Vector3dVector(frame_voxels_xyz)
            frame_pcd1.colors = o3d.utility.Vector3dVector(frame_voxels_color / 255)
            o3d.io.write_point_cloud(str(output_dir / f'{frame_key}.ply'), frame_pcd1)
        # <================ debug

        # 世界坐标内画出场景的范围
        xs = np.arange(voxOriginWorld[0, 0], voxOriginWorld[0, 0] + 100 * voxel_size, voxel_size)[:voxDim[0]]
        ys = np.arange(voxOriginWorld[1, 0], voxOriginWorld[1, 0] + 100 * voxel_size, voxel_size)[:voxDim[1]]
        zs = np.arange(voxOriginWorld[2, 0], voxOriginWorld[2, 0] + 100 * voxel_size, voxel_size)[:voxDim[2]]
        gridPtsWorldX, gridPtsWorldY, gridPtsWorldZ = np.meshgrid(xs, ys, zs)
        gridPtsWorld = np.stack([gridPtsWorldX.flatten(),
                                 gridPtsWorldY.flatten(),
                                 gridPtsWorldZ.flatten()], axis=1)
        gridPtsLabel = np.zeros((gridPtsWorld.shape[0]))
        gridPtsWorld_color = np.zeros((gridPtsWorld.shape[0], 3))

        # ================> debug
        if vis_debug:
            frame_pcd_with_grid = o3d.geometry.PointCloud()
            frame_voxels_with_grid_xyz = np.vstack((frame_voxels, gridPtsWorld))
            frame_voxels_with_grid_color = np.vstack((color_map[scene_voxels_sem[mask]], gridPtsWorld_color))
            frame_pcd_with_grid.points = o3d.utility.Vector3dVector(frame_voxels_with_grid_xyz)
            frame_pcd_with_grid.colors = o3d.utility.Vector3dVector(frame_voxels_with_grid_color / 255)
            o3d.io.write_point_cloud(str(output_dir / f'{frame_key}_world.ply'), frame_pcd_with_grid)
        # <================ debug
        kdtree = KDTree(frame_voxels[:, :3], leaf_size=10)
        dist, ind = kdtree.query(gridPtsWorld)  # 返回与gridPtsWorld最近的1个邻居。dist表示其距离，ind表示其索引
        dist, ind = dist.reshape(-1), ind.reshape(-1)
        mask = dist <= voxel_size  # 确保最近的邻居，在范围之内
        gridPtsLabel[mask] = frame_voxels_sem[ind[mask]]  # 赋予语义标签

        g = gridPtsLabel.reshape(voxDim[0], voxDim[1], voxDim[2])
        g_not_0 = np.where(g > 0)  # 初始化是0
        if len(g_not_0) == 0:
            continue
        g_not_0_x = g_not_0[0]
        g_not_0_y = g_not_0[1]
        if len(g_not_0_x) == 0:
            continue
        if len(g_not_0_y) == 0:
            continue
        valid_x_min = g_not_0_x.min()
        valid_x_max = g_not_0_x.max()
        valid_y_min = g_not_0_y.min()
        valid_y_max = g_not_0_y.max()
        mask = np.zeros_like(g)
        if valid_x_min != valid_x_max and valid_y_min != valid_y_max:
            mask[valid_x_min:valid_x_max, valid_y_min:valid_y_max, :] = 1
            mask = 1 - mask  #
            mask = mask.astype(np.bool_)
            g[mask] = 255  # 在有效范围以外的区域, 将其label设置为255
        else:
            continue
        frame_voxels = np.zeros((gridPtsWorld.shape[0], 4))
        frame_voxels[:, :3] = gridPtsWorld
        frame_voxels[:, -1] = g.reshape(-1)
        # gridPtsWorld[:, -1] = g.reshape(-1)

        # 计算3D点至2D图像的投影点
        voxels_cam = (np.linalg.inv(cam2world)[:3, :3] @ gridPtsWorld[:, :3].T \
                      + np.linalg.inv(cam2world)[:3, -1:]).T
        voxels_pix = (intrinsics[:3, :3] @ voxels_cam.T).T
        voxels_pix = voxels_pix / voxels_pix[:, -1:]
        mask = np.logical_and(voxels_pix[:, 0] >= 0,
                              np.logical_and(voxels_pix[:, 0] < w,
                                             np.logical_and(voxels_pix[:, 1] >= 0,
                                                            np.logical_and(voxels_pix[:, 1] < h,
                                                                           voxels_cam[:, 2] > 0))))  # 视野内的
        inroom = frame_voxels[:, -1] != 255
        mask = np.logical_and(~mask, inroom)  # 如果一个3d point，它没有落在图像上，并且是在房间内，则将其label设置为0（empty）
        frame_voxels[mask, -1] = 0  # empty类别

        if vis_debug:
            # 仅保存有效的语义
            valid_frame_voxels = frame_voxels[np.logical_and(frame_voxels[:, -1] > 0, frame_voxels[:, -1] < 13)]
            frame_pcd2 = o3d.geometry.PointCloud()
            frame_pcd2.points = o3d.utility.Vector3dVector(valid_frame_voxels[:, :3])
            frame_pcd2.colors = o3d.utility.Vector3dVector(color_map[valid_frame_voxels[:, -1].astype(int)] / 255)
            o3d.io.write_point_cloud(str(output_dir / f'{frame_key}_valid.ply'), frame_pcd2)

            # 保留empty区域+语义区域
            valid_frame_voxels = frame_voxels[frame_voxels[:, -1] < 13]
            frame_pcd3 = o3d.geometry.PointCloud()
            frame_pcd3.points = o3d.utility.Vector3dVector(valid_frame_voxels[:, :3])
            frame_pcd3.colors = o3d.utility.Vector3dVector(color_map[valid_frame_voxels[:, -1].astype(int)] / 255)
            o3d.io.write_point_cloud(str(output_dir / f'{frame_key}_valid+empty.ply'), frame_pcd3)

            # 保留unknown区域
            unknown_frame_voxels = frame_voxels[frame_voxels[:, -1] == 255]
            frame_pcd4 = o3d.geometry.PointCloud()
            frame_pcd4.points = o3d.utility.Vector3dVector(unknown_frame_voxels[:, :3])
            frame_pcd4.colors = o3d.utility.Vector3dVector(color_map[unknown_frame_voxels[:, -1].astype(int)] / 255)
            o3d.io.write_point_cloud(str(output_dir / f'{frame_key}_unknown.ply'), frame_pcd4)

            # unknown+sem区域
            unknown_sem_frame_voxels = frame_voxels[frame_voxels[:, -1] > 0]
            # sem_frame_voxels = frame_voxels[np.logical_and(frame_voxels[:, -1] > 0, frame_voxels[:, -1] < 13)]
            frame_pcd4 = o3d.geometry.PointCloud()
            frame_pcd4.points = o3d.utility.Vector3dVector(unknown_sem_frame_voxels[:, :3])
            frame_pcd4.colors = o3d.utility.Vector3dVector(color_map[unknown_sem_frame_voxels[:, -1].astype(int)] / 255)
            o3d.io.write_point_cloud(str(output_dir / f'{frame_key}_unknown_sem.ply'), frame_pcd4)
        #######

        # ===================================================================
        # 2025-06-26
        # <<< 可见性过滤 >>>
        # ===================================================================
        # 在这里，`final_labeled_voxels` 包含了当前帧附近的所有体素及其初步标签 (0=empty, 1-12=sem, 255=unknown)
        # 我们现在要基于深度图，把被遮挡的体素找出来，并把它们的标签设为0 (empty)

        # 1. 获取所有体素点的世界坐标和当前标签
        voxel_coords = frame_voxels[:, :3]
        voxel_labels = frame_voxels[:, -1]

        # 2. 将所有体素点转换到相机坐标系
        world2cam = np.linalg.inv(cam2world)
        voxels_cam = (world2cam[:3, :3] @ voxel_coords.T + world2cam[:3, -1:]).T

        # 3. 投影到像素平面
        voxels_pix = (intrinsics[:3, :3] @ voxels_cam.T).T

        depths_in_cam = voxels_cam[:, 2]
        # 防止除以零
        depths_in_cam[depths_in_cam <= 0] = 1e6  # 把相机后方的点深度设为极大值，使其在后续比较中被自然剔除

        us = (voxels_pix[:, 0] / depths_in_cam).astype(int)
        vs = (voxels_pix[:, 1] / depths_in_cam).astype(int)

        # 4. 创建一个遮挡掩码，初始假设所有点都未被遮挡
        # 我们只关心那些有标签的体素（非empty, 非unknown）
        occlusion_mask = np.zeros_like(voxel_labels, dtype=bool)

        # 5. 筛选出需要进行深度测试的体素
        # 条件：a. 在图像范围内 b. 在相机前方 c. 有一个有效的语义标签 (不是empty也不是unknown)
        test_indices = np.where(
            (us >= 0) & (us < w) &
            (vs >= 0) & (vs < h) &
            (depths_in_cam > 0) &
            (voxel_labels > 0) & (voxel_labels != 255)
        )[0]

        if len(test_indices) > 0:
            # 获取这些待测点的像素坐标和计算深度
            test_us = us[test_indices]
            test_vs = vs[test_indices]
            test_depths = depths_in_cam[test_indices]

            # 获取深度图中对应位置的深度值
            gt_depth_values = depth[test_vs, test_us]

            # 识别被遮挡的点：其计算深度显著大于GT深度
            is_occluded = test_depths > (gt_depth_values + voxel_size * 5)

            # 将被遮挡的点的索引更新到遮挡掩码中
            occluded_indices = test_indices[is_occluded]
            occlusion_mask[occluded_indices] = True

        # 6. 应用遮挡掩码：将被遮挡的体素标签设为0 (empty)
        final_labels = voxel_labels.copy()
        final_labels[occlusion_mask] = 0
        frame_voxels[:, -1] = final_labels

        # [Debug Visualization - 可选]
        if vis_debug:
            # 可视化过滤后的结果
            visible_voxels = np.hstack((gridPtsWorld, final_labels.reshape(-1, 1)))
            visible_voxels = visible_voxels[np.logical_and(visible_voxels[:, -1] > 0, visible_voxels[:, -1] < 13)]

            if len(visible_voxels) > 0:
                frame_pcd_filtered = o3d.geometry.PointCloud()
                frame_pcd_filtered.points = o3d.utility.Vector3dVector(visible_voxels[:, :3])
                frame_pcd_filtered.colors = o3d.utility.Vector3dVector(
                    color_map[visible_voxels[:, -1].astype(int)] / 255)
                o3d.io.write_point_cloud(str(output_dir / f'{frame_key}_final_visible.ply'), frame_pcd_filtered)

        # ===================================================================
        # <<<  新的最终可见性过滤 >>>
        # ===================================================================

        ###### 保存
        target_1_4 = frame_voxels[:, -1].reshape(60, 60, 36)

        pkl_data = {
            'img': str(input_color_dir / f'{frame_key}.jpg'),
            'depth_gt': str(input_depth_dir / f'{frame_key}.png'),
            'cam_pose': pose,  # camera to world
            'intrinsic': intrinsics,
            'target_1_4': target_1_4,  # 1_4 表示下采样了4倍, 8cm
            'voxel_origin': np.array([frame_voxels[:, 0].min(), frame_voxels[:, 1].min(), frame_voxels[:, 2].min()]),
        }
        with open(str(output_dir / f'{frame_key}.pkl'), "wb") as handle:
            pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
        '--input_mesh',
        type=str,
        default='point_lifted_mesh.ply',
        help='Name of input mesh',
    )
    parser.add_argument(
        '--input_label',
        type=str,
        default='labels.txt',
        help='Name of input label for 3d mesh',
    )  # 每个点的label
    parser.add_argument(
        '--output_global_occupancy',
        type=str,
        default='occupancy.ply',
        help='Name of files to save the occupancy',
    )
    parser.add_argument(
        '--label_space',
        default='occ11'
    )  # ['wordnet','occ11']

    parser.add_argument(
        '--output_frame_folder',
        default='preprocessed_voxels'
    )

    parser.add_argument(
        '--config',
        help='Name of config file'
    )

    return parser.parse_args()


if __name__ == '__main__':
    """
        Convert scene mesh to scene occupancy
    """

    args = arg_parser()
    if args.config is not None:
        gin.parse_config_file(args.config)
    main(
        scene_dir=args.workspace,
        input_label=args.input_label,
        input_mesh=args.input_mesh,
        output_global_occupancy=args.output_global_occupancy,
        output_frame_folder=args.output_frame_folder,
        label_space=args.label_space,
    )
