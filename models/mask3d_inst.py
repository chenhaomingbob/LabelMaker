import argparse
import logging
import os
import random
import shutil
from copy import deepcopy
from os.path import abspath, dirname, join, relpath
from pathlib import Path
from typing import Union

import albumentations as A
import cv2
import gin
import mask3d.conf
import MinkowskiEngine as ME
import numpy as np
import open3d as o3d
import torch
import torch.backends.cudnn as cudnn
from hydra.experimental import compose, initialize
# from hydra import initialize,compose
from mask3d import InstanceSegmentation
from mask3d.datasets.scannet200.scannet200_constants import SCANNET_COLOR_MAP_200, VALID_CLASS_IDS_200, CLASS_LABELS_200
from mask3d.utils.utils import (
    load_backbone_checkpoint_with_missing_or_exsessive_keys,
    load_checkpoint_with_missing_or_exsessive_keys,
)
from torch.nn.functional import softmax
from tqdm import tqdm

from labelmaker.label_data import get_ade150

logging.basicConfig(level="INFO")
logger = logging.getLogger('Mask 3D Mesh preprocessing')

INSTANCE_COLOR_MAP = {
    0: (255, 0, 0),
    1: (255, 128, 0),
    2: (255, 255, 0),
    3: (128, 255, 0),
    4: (0, 255, 0),
    5: (0, 255, 128),
    6: (0, 255, 255),
    7: (0, 128, 255),
    8: (0, 0, 255),
    9: (128, 0, 255),
    10: (255, 0, 128),
    11: (255, 0, 255),
    12: (128, 0, 128),
    13: (192, 192, 192),
    14: (128, 128, 128),
    15: (255, 99, 71),
    16: (255, 153, 0),
    17: (255, 204, 0),
    18: (204, 255, 0),
    19: (153, 255, 0),
    20: (0, 255, 153),
    21: (0, 255, 204),
    22: (0, 204, 255),
    23: (0, 153, 255),
    24: (0, 102, 255),
    25: (0, 51, 255),
    26: (0, 255, 51),
    27: (0, 255, 102),
    28: (0, 255, 153),
    29: (0, 255, 204),
    30: (0, 255, 255),
    31: (51, 0, 255),
    32: (102, 0, 255),
    33: (153, 0, 255),
    34: (204, 0, 255),
    35: (255, 0, 204),
    36: (255, 0, 153),
    37: (255, 0, 102),
    38: (255, 0, 51),
    39: (255, 66, 0),
    40: (255, 102, 0),
    41: (255, 153, 0),
    42: (255, 204, 0),
    43: (255, 255, 0),
    44: (204, 255, 0),
    45: (153, 255, 0),
    46: (102, 255, 0),
    47: (51, 255, 0),
    48: (0, 255, 51),
    49: (0, 255, 102),
    50: (0, 255, 153),
    51: (0, 255, 204),
    52: (0, 255, 255),
    53: (0, 204, 255),
    54: (0, 153, 255),
    55: (0, 102, 255),
    56: (0, 51, 255),
    57: (0, 0, 255),
    58: (51, 0, 255),
    59: (102, 0, 255),
    60: (153, 0, 255),
}


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_config_and_model(checkpoint_path: str):
    conf_path = relpath(
        dirname(mask3d.conf.__file__),
        start=abspath(dirname(__file__)),
        # start=abspath(relpath(dirname(__file__))),
    )
    with initialize(config_path=conf_path):
        cfg = compose(config_name="config_base_instance_segmentation.yaml")

    # these are copied from Francis's demo code
    cfg.general.checkpoint = checkpoint_path
    cfg.general.dbscan_eps = 0.95
    cfg.general.num_targets = 201
    cfg.general.scores_threshold = 0.1
    cfg.general.train_mode = False
    cfg.general.topk_per_image = 300
    cfg.general.use_dbscan = False
    cfg.data.num_labels = 200
    cfg.data.test_mode = "test"
    cfg.model.num_queries = 150
    model = InstanceSegmentation(cfg)

    if cfg.general.backbone_checkpoint is not None:
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(
            cfg, model)
    if cfg.general.checkpoint is not None:
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    return cfg, model


def get_mask_and_scores(
        mask_cls: torch.Tensor,
        mask_pred: torch.Tensor,
        topk_per_image: int,
        num_queries: int,
        num_classes: int,
        device: Union[str, torch.device],
):
    """
      The logic follows from mask3d.trainer.trainer.InstancSegmentation.get_mask_and_scores
    """
    labels = torch.arange(
        num_classes,
        device=device,
    ).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)

    if topk_per_image != -1:
        scores_per_query, topk_indices = mask_cls.flatten(0, 1).topk(
            topk_per_image,
            sorted=True,
        )
    else:
        scores_per_query, topk_indices = mask_cls.flatten(0, 1).topk(
            num_queries,
            sorted=True,
        )

    labels_per_query = labels[topk_indices]
    topk_indices = topk_indices // num_classes
    mask_pred = mask_pred[:, topk_indices]

    result_pred_mask = (mask_pred > 0).float()
    heatmap = mask_pred.float().sigmoid()

    mask_scores_per_image = (heatmap * result_pred_mask).sum(0) / (
            result_pred_mask.sum(0) + 1e-6)
    score = scores_per_query * mask_scores_per_image
    classes = labels_per_query

    return score, result_pred_mask, classes, heatmap


def run_mask3d(
        scene_dir: Union[str, Path],
        output_folder: Union[str, Path],
        device: Union[str, torch.device] = 'cuda:0',
        flip: bool = False,
):
    scene_dir = Path(scene_dir)
    output_folder = Path(output_folder)

    assert scene_dir.exists() and scene_dir.is_dir()

    model_ckpt = abspath(
        join(__file__, '../../checkpoints/mask3d_scannet200_demo.ckpt'))
    cfg, model = get_config_and_model(checkpoint_path=model_ckpt)
    model = model.to(device).eval()

    color_mean = (0.47793125906962, 0.4303257521323044, 0.3749598901421883)
    color_std = (0.2834475483823543, 0.27566157565723015, 0.27018971370874995)
    normalize_color = A.Normalize(mean=color_mean, std=color_std)

    # load point cloud
    input_mesh_path = str(scene_dir / 'mesh.ply')
    mesh = o3d.io.read_triangle_mesh(input_mesh_path)

    points = np.asarray(mesh.vertices).copy()
    if flip:
        points[:, 0] *= -1  # flip x axis
    colors = np.asarray(mesh.vertex_colors)
    colors = colors * 255.

    pseudo_image = colors.astype(np.uint8)[np.newaxis, :, :]
    colors = np.squeeze(normalize_color(image=pseudo_image)["image"])

    coords = np.floor(points / 0.02)
    _, _, unique_map, inverse_map = ME.utils.sparse_quantize(
        coordinates=coords,
        features=colors,
        return_index=True,
        return_inverse=True,
    )

    sample_coordinates = coords[unique_map]
    coordinates = [torch.from_numpy(sample_coordinates).int()]
    sample_features = colors[unique_map]
    features = [torch.from_numpy(sample_features).float()]

    coordinates, _ = ME.utils.sparse_collate(coords=coordinates, feats=features)
    features = torch.cat(features, dim=0)
    data = ME.SparseTensor(
        coordinates=coordinates,
        features=features,
        device=device,
    )

    # run model
    with torch.no_grad():
        outputs = model(data, raw_coordinates=features)

    del data
    torch.cuda.empty_cache()

    # parse predictions
    mask_cls = softmax(outputs["pred_logits"][0], dim=-1)[..., :-1].detach().cpu()
    mask_pred = outputs["pred_masks"][0].detach().cpu()

    scores, masks, classes, _ = get_mask_and_scores(
        mask_cls=mask_cls,
        mask_pred=mask_pred,
        topk_per_image=cfg.general.topk_per_image,
        num_queries=cfg.model.num_queries,
        num_classes=model.model.num_classes - 1,
        device=device,
    )

    # sort according to score
    sorted_scores = scores.sort(descending=True)
    sorted_scores_index = sorted_scores.indices.cpu().numpy()
    sorted_scores_values = sorted_scores.values.cpu().numpy()
    sorted_classes = classes[sorted_scores_index]
    sorted_masks = masks[:, sorted_scores_index].numpy()

    # filter_out_instance, this is in InstanceSegmentation.eval_instance_step
    keep_instances = set()
    pairwise_overlap = sorted_masks.T @ sorted_masks
    normalization = pairwise_overlap.max(axis=0)
    norm_overlaps = pairwise_overlap / (normalization + 1e-8)

    for instance_id in range(norm_overlaps.shape[0]):
        if not (sorted_scores_values[instance_id] < cfg.general.scores_threshold):
            # check if mask != empty
            if not sorted_masks[:, instance_id].sum() == 0.0:
                overlap_ids = set(
                    np.nonzero(
                        norm_overlaps[instance_id, :] > cfg.general.iou_threshold)[0])

                if len(overlap_ids) == 0:
                    keep_instances.add(instance_id)
                else:
                    if instance_id == min(overlap_ids):
                        keep_instances.add(instance_id)

    keep_instances = sorted(list(keep_instances))

    # label processing, wall and floor are ignored
    modified_classes = sorted_classes[keep_instances] + 2
    modified_classes[modified_classes == 2] = 1

    filtered_classes = modified_classes.tolist()
    filtered_scores = sorted_scores_values[keep_instances].tolist()
    filtered_masks_binary = [
        sorted_masks[:, idx][inverse_map] > 0.5 for idx in keep_instances
    ]

    # save labelled mesh
    mesh_labelled = o3d.geometry.TriangleMesh()
    mesh_labelled.vertices = mesh.vertices
    mesh_labelled.triangles = mesh.triangles

    labels_mapped = np.zeros((len(mesh.vertices), 1))
    colors_mapped = np.zeros((len(mesh.vertices), 3))

    # 根据语义保存
    for label, mask in zip(filtered_classes, filtered_masks_binary):
        labels_mapped[mask] = label
        colors_mapped[mask] = SCANNET_COLOR_MAP_200[VALID_CLASS_IDS_200[label]]

    output_dir = scene_dir / output_folder
    output_dir = Path(str(output_dir) + '_flip') if flip else output_dir
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(str(output_dir), exist_ok=False)

    # saving ply for semantic segmentation
    mesh_labelled.vertex_colors = o3d.utility.Vector3dVector(
        colors_mapped.astype(np.float32) / 255.)
    o3d.io.write_triangle_mesh(
        f'{str(output_dir)}/mesh_labelled.ply',
        mesh_labelled,
    )
    # from models.cad_utils.instance_segmentation_to_cad_format import convert_to_cad_format
    ######################## 2025-05-29 ########################
    # 根据实例保存
    # convert_to_cad_format(
    #     scene_mesh=mesh,
    #     object_classes=filtered_classes,
    #     object_masks=filtered_masks_binary,
    #     output_dir=output_dir
    # )
    # inst_labels_mapped = np.zeros((len(mesh.vertices), 1))
    # inst_colors_mapped = np.zeros((len(mesh.vertices), 3))
    # all_boxes_selected = None
    # for inst_id, (label, mask) in enumerate(zip(filtered_classes, filtered_masks_binary)):
    #     inst_labels_mapped[mask] = inst_id
    #     inst_colors_mapped[mask] = INSTANCE_COLOR_MAP[inst_id]
    #     ###
    #     instance_vertices = np.asarray(mesh.vertices)[mask]  # instance points
    #     # 过滤一些类别
    #     class_name = CLASS_LABELS_200[label]
    #
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(instance_vertices)
    #     o3d.io.write_point_cloud(f"{inst_id}.ply", pcd)
    #
    #     # pcd=o3d.geometry.TriangleMesh()
    #     # pcd.vertices = o3d.utility.Vector3dVector(instance_vertices)
    #     # o3d.io.write_triangle_mesh(f"{inst_id}.ply", pcd)
    #     o3d.visualization.draw_geometries([pcd], window_name=f"{class_name}")
    #     # obb
    #     bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(instance_vertices))
    #
    #     box_final = get_corners_of_bb3d_no_index(bbox.R.T, bbox.extent / 2, bbox.center)  # 计算bbox的8个顶点
    #     # center_final, basis_final, coeffs_final = get_bdb_from_corners(box_final)
    #     # box_final = get_corners_of_bb3d_no_index(basis_final.T, coeffs_final, center_final)
    #
    #     line_color = [0, 1, 0]
    #     lineSets_selected = drawOpen3dCylLines([box_final], line_color)
    #     # o3d.io.write_triangle_mesh(f"{inst_id}_with_bbox.ply", lineSets_selected + pcd)
    #     # o3d.visualization.draw_geometries([pcd, lineSets_selected], window_name="Mesh and PointCloud")
    #     if all_boxes_selected is None:
    #         all_boxes_selected = lineSets_selected
    #     else:
    #         all_boxes_selected += lineSets_selected
    #
    # o3d.io.write_triangle_mesh("combined_mesh_with_obb.ply", all_boxes_selected + mesh)
    #
    # # saving ply for instance segmentation
    # mesh_inst_labelled = o3d.geometry.TriangleMesh()
    # mesh_inst_labelled.vertices = mesh.vertices
    # mesh_inst_labelled.triangles = mesh.triangles
    # mesh_inst_labelled.vertex_colors = o3d.utility.Vector3dVector(
    #     inst_colors_mapped.astype(np.float32) / 255.)  # 顶点的颜色
    # o3d.io.write_triangle_mesh(
    #     f'{str(output_dir)}/mesh_inst_labelled.ply',
    #     mesh_inst_labelled,
    # )
    ######################## 2025-05-29 ############################

    mask_path = output_dir / 'pred_mask'
    mask_path.mkdir(exist_ok=True)
    with open(str(output_dir / 'predictions.txt'), 'w') as f:
        for i, (label, score, mask) in tqdm(
                enumerate(zip(
                    filtered_classes,
                    filtered_scores,
                    filtered_masks_binary,
                )),
                total=len(filtered_classes),
        ):
            mask_file = f'pred_mask/{str(i).zfill(3)}.txt'
            f.write(f'{mask_file} {VALID_CLASS_IDS_200[label]} {score}\n')  # 写到
            np.savetxt(
                f'{str(output_dir)}/pred_mask/{str(i).zfill(3)}.txt',
                mask,
                fmt='%d',
            )


def run_rendering(
        scene_dir: Union[str, Path],
        output_folder: Union[str, Path],
        resolution=(480, 640),
        flip: bool = False,
):
    scene_dir = Path(scene_dir)
    output_folder = Path(output_folder)

    assert scene_dir.exists() and scene_dir.is_dir()

    input_pose_dir = scene_dir / 'pose'
    assert input_pose_dir.exists() and input_pose_dir.is_dir()

    output_dir = scene_dir / output_folder
    output_dir = Path(str(output_dir) + '_flip') if flip else output_dir

    prediction_file = output_dir / 'predictions.txt'
    if not prediction_file.exists():
        logger.error(f'No prediction file found in {scene_dir}')
        return

    with open(prediction_file) as f:
        instances = [x.strip().split(' ') for x in f.readlines()]

    # read mesh
    mesh_path = scene_dir / "mesh.ply"
    assert mesh_path.exists()
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))

    labelinfo = get_ade150()
    colors = np.array([x['color'] for x in labelinfo])
    colors = colors / 255.0

    objects = []
    scenes = []
    # render = o3d.visualization.rendering.OffscreenRenderer(640, 480)
    geoid_to_classid = {}

    for i, inst in enumerate(instances):
        # check confidence
        if float(inst[2]) < 0.5:
            continue
        scene = o3d.t.geometry.RaycastingScene()
        filepath = output_dir / inst[0]
        mask = np.loadtxt(filepath).astype(bool)
        obj = deepcopy(mesh)
        obj.remove_vertices_by_mask(np.logical_not(mask))
        obj.paint_uniform_color(colors[int(inst[1]) % 150])
        # obj.paint_uniform_color((0.5, 0.5, 0.00001 * int(inst[1])))
        objects.append(obj)
        obj_in_scene = o3d.t.geometry.TriangleMesh.from_legacy(obj)

        geoid_to_classid[i] = int(inst[1])
        # render.scene.add_geometry(f"object{i}", obj, materials[int(inst[1])])
        scene.add_triangles(obj_in_scene)
        scenes.append(scene)

    # o3d.visualization.draw_geometries(objects)

    keys = [x.stem for x in scene_dir.glob('pose/*.txt')]
    for k in tqdm(keys):
        cam_to_world = np.loadtxt(scene_dir / 'pose' / f'{k}.txt')
        world_to_cam = np.eye(4)
        world_to_cam[:3, :3] = cam_to_world[:3, :3].T
        world_to_cam[:3, 3] = -world_to_cam[:3, :3] @ cam_to_world[:3, 3]

        intrinsics = np.loadtxt(scene_dir / 'intrinsic' / f'{k}.txt')

        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            width_px=resolution[1],
            height_px=resolution[0],
            intrinsic_matrix=intrinsics[:3, :3],
            extrinsic_matrix=world_to_cam,  # world to camera
        )
        pixelid_to_instance = []
        segmentation = -1 * np.ones(resolution).astype(int)
        rendered_distance = np.zeros(resolution)
        for i, scene in enumerate(scenes):
            vis = scene.cast_rays(rays)
            geometry_ids = vis['geometry_ids'].numpy().astype(int)
            pixelid_to_instance.append([i])
            mask = geometry_ids == 0
            # check if this instance occludes a previous instance
            occluding_previous_pred = np.logical_and(
                rendered_distance > vis['t_hit'].numpy() + 0.05, mask)
            segmentation[occluding_previous_pred] = len(pixelid_to_instance) - 1
            rendered_distance[occluding_previous_pred] = vis['t_hit'].numpy(
            )[occluding_previous_pred]
            mask = np.logical_and(mask, np.logical_not(occluding_previous_pred))
            # now check if this instance gets occluded
            occluded_by_previous_pred = np.logical_and(
                rendered_distance <= vis['t_hit'].numpy() + 0.05, rendered_distance
                != 0)
            mask[occluded_by_previous_pred] = False
            # now deal with the case where there is no overlap with other ids
            update_mask = np.logical_and(mask, segmentation == -1)
            segmentation[update_mask] = len(pixelid_to_instance) - 1
            rendered_distance[update_mask] = vis['t_hit'].numpy()[update_mask]
            mask[update_mask] = False
            # finally, there are cases where already another instance was rendered at the same position
            for overlapping_id in np.unique(segmentation[np.logical_and(
                    mask, segmentation != -1)]):
                # check if this already overlaps with something else
                if len(pixelid_to_instance[overlapping_id]) > 1:
                    # merge
                    pixelid_to_instance[overlapping_id] = list(
                        set(pixelid_to_instance[overlapping_id] + [i]))
                else:
                    # new multi-instance
                    pixelid_to_instance.append(
                        [pixelid_to_instance[overlapping_id][0], i])
                    segmentation[np.logical_and(
                        mask,
                        segmentation == overlapping_id)] = len(pixelid_to_instance) - 1

        semantic_segmentation = np.zeros(resolution).astype(np.uint16)
        for i, ids in enumerate(pixelid_to_instance):
            if len(ids) == 1:
                semantic_segmentation[segmentation == i] = instances[ids[0]][1]
            else:
                max_confidence = -1
                max_id = 0  # default is the unknown class
                for j in ids:
                    if float(instances[j][2]) > max_confidence:
                        max_confidence = float(instances[j][2])
                        max_id = instances[j][1]
                semantic_segmentation[segmentation == i] = max_id

        cv2.imwrite(str(output_dir / f'{k}.png'), semantic_segmentation)


@gin.configurable
def run(
        scene_dir: Union[str, Path],
        output_folder: Union[str, Path],
        device: Union[str, torch.device] = 'cuda:0',
        render_resolution=(480, 640),
        flip: bool = False,
):
    run_mask3d(
        scene_dir=scene_dir,
        output_folder=output_folder,
        device=device,
        flip=flip,
    )
    run_rendering(
        scene_dir=scene_dir,
        output_folder=output_folder,
        resolution=render_resolution,
        flip=flip,
    )


def arg_parser():
    parser = argparse.ArgumentParser(description='Mask3D Segmentation')
    parser.add_argument(
        '--workspace',
        type=str,
        required=True,
        help=
        'Path to workspace directory. There should be a "mesh.ply" file and "pose" folder inside.',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='intermediate/scannet200_mask3d_1',
        help=
        'Name of output directory in the workspace directory intermediate. Has to follow the pattern $labelspace_$model_$version.'
    )
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--config', help='Name of config file')
    parser.add_argument(
        '--flip',
        action="store_true",
        help='Mirror the input mesh file, this is part of test time augmentation.',
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    if args.config is not None:
        gin.parse_config_file(args.config)

    setup_seeds(seed=args.seed)
    run(scene_dir=args.workspace, output_folder=args.output, flip=args.flip)
