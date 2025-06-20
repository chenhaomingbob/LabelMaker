import os
import pickle
import numpy as np
import open3d as o3d
from .scannet_annotation import ScanNetAnnotation,ObjectAnnotation
from .obb_utils import drawOpen3dCylLines, get_corners_of_bb3d_no_index

from mask3d.datasets.scannet200.scannet200_constants import SCANNET_COLOR_MAP_200, VALID_CLASS_IDS_200, CLASS_LABELS_200


# def parse_cls_label(cls_label_scannetpp):
#     cls_label_shapenet = None
#
#     if cls_label_scannetpp in ['table', 'desk', 'office table', 'rolling table', 'experiment bench', 'laboratory bench',
#                                'office desk', 'work bench', 'coffee table', 'conference table', 'dining table',
#                                'high table', 'ping pong table', 'bedside table', 'computer desk', 'foosball table',
#                                'babyfoot table',
#                                'side table', 'workbench', 'table football', 'trolley table', 'serving trolley',
#                                'bar table', 'computer table', 'tv table', 'work station', 'study table', 'footstool',
#                                'tv console', 'tv trolley',
#                                'center table', 'folded table', 'short table', 'sidetable', 'folding table',
#                                'laptop table', 'joined tables']:
#         cls_label_shapenet = 'table'
#     elif cls_label_scannetpp in ['standing lamp', 'floor lamp', 'light', 'table lamp',
#                                  'lamp', 'desk lamp', 'studio light', 'bedside lamp',
#                                  'ring light', 'door lamp', 'desk light', 'monitor light',
#                                  'kitchen light']:
#         cls_label_shapenet = 'lamp'
#     elif cls_label_scannetpp in ['chair', 'office chair', 'stool', 'sofa chair', 'beanbag', 'arm chair', 'lounge chair',
#                                  'armchair', 'dining chair', 'office visitor chair', 'seat', 'chairs', 'rolling chair',
#                                  'high stool',
#                                  'wheelchair', 'ottoman chair', 'recliner', 'barber chair', 'papasan chair',
#                                  'toilet seat', 'easy chair', 'step stool', 'deck chair', 'high chair', 'office  chair',
#                                  'medical stool',
#                                  'barstool', 'linked retractable seats', 'folding chair']:
#         cls_label_shapenet = 'chair'
#     elif cls_label_scannetpp in ['wall cabinet', 'locker', 'storage cabinet', 'cabinet', 'kitchen cabinet', 'wardrobe',
#                                  'cupboard', 'office cabinet', 'laboratory cabinet', 'kitchen unit', 'file cabinet',
#                                  'open cabinet', 'closet', 'bathroom cabinet', 'bath cabinet',
#                                  'clothes cabinet', 'fitted wardrobe',
#                                  'small cabinet', 'bottle crate', 'foldable closet', 'tool rack',
#                                  'power cabinet', 'shoe cabinet', 'drawer', 'bedside cabinet',
#                                  'shelf trolley', 'wall unit',
#                                  'switchboard cabinet', 'bedside shelf', 'file storage', 'shoe box', 'mirror cabinet',
#                                  'first aid cabinet', 'mobile tv stand', 'tv stand', 'television stand',
#                                  'monitor stand']:
#         cls_label_shapenet = 'cabinet'
#     elif cls_label_scannetpp in ['bookshelf', 'storage rack', 'shoe rack', 'book shelf', 'storage shelf',
#                                  'kitchen shelf', 'bathroom shelf', 'shoes holder', 'garage shelf',
#                                  'kitchen storage rack', 'bathroom rack',
#                                  'glass shelf', 'clothes rack', 'shower rug', 'wall shelf', 'recesssed shelf',
#                                  'recessed shelve', 'shelve', 'wine rack']:
#         cls_label_shapenet = 'bookshelf'
#     elif cls_label_scannetpp in ['monitor', 'tv', 'projector screen', 'television', 'flat panel display',
#                                  'tv screen', 'tv mount', 'computer monitor',
#                                  'screen']:
#         cls_label_shapenet = 'display'
#     elif cls_label_scannetpp in ['oven', 'stove', 'oven range', 'kitchen stove']:
#         cls_label_shapenet = 'stove'
#     elif cls_label_scannetpp in ['sofa', 'couch', 'l-shaped sofa', 'floor sofa', 'folding sofa', 'floor couch']:
#         cls_label_shapenet = 'sofa'
#     elif cls_label_scannetpp in ['bed', 'loft bed', 'canopy bed', 'camping bed']:
#         cls_label_shapenet = 'bed'
#     elif cls_label_scannetpp in ['pillow', 'cushion', 'floor cushion', 'sit-up pillow', 'sofa cushion', 'long pillow',
#                                  'seat cushion', 'chair cushion']:
#         cls_label_shapenet = 'pillow'
#     elif cls_label_scannetpp in ['plant pot', 'pot', 'vase', 'flower pot']:
#         cls_label_shapenet = 'flowerpot'
#     elif cls_label_scannetpp in ['trash can', 'trash bin', 'bucket', 'bin', 'trashcan', 'dustbin', 'jerry can',
#                                  'garbage bin', 'storage bin', 'wastebin', 'recycle bin']:
#         cls_label_shapenet = 'trash bin'
#     elif cls_label_scannetpp in ['sink', 'kitchen sink', 'bathroom sink', 'shower sink']:
#         cls_label_shapenet = 'bathtub'
#     elif cls_label_scannetpp in ['suitcase', 'backpack', 'bagpack', 'luggage bag', 'suit bag', 'rucksack',
#                                  'package bag']:
#         cls_label_shapenet = 'bag'
#     elif cls_label_scannetpp in ['refrigerator', 'fridge', 'mini fridge', 'lab fridge', 'freezer']:
#         cls_label_shapenet = 'refridgerator'
#     elif cls_label_scannetpp in ['bathtub', 'bath tub', 'basin', 'washbasin', 'wash basin']:
#         cls_label_shapenet = 'bathtub'
#     elif cls_label_scannetpp in ['keyboard']:
#         cls_label_shapenet = 'keyboard'
#     elif cls_label_scannetpp in ['toilet', 'urinal']:
#         cls_label_shapenet = 'toilet'
#     elif cls_label_scannetpp in ['printer', 'photocopy machine', 'copier', '3d printer', 'overhead projector',
#                                  'copy machine', 'paper shredder', 'multifunction printer', 'photocopier',
#                                  'scanner']:
#         cls_label_shapenet = 'printer'
#     elif cls_label_scannetpp in ['bench', 'foot rest', 'piano stool', 'bench stool', 'shoe stool', 'piano chair',
#                                  'high bench']:
#         cls_label_shapenet = 'bench'
#     elif cls_label_scannetpp in ['microwave', 'microwave oven', 'toaster oven', 'mini oven']:
#         cls_label_shapenet = 'microwaves'
#     elif cls_label_scannetpp in ['basket', 'laundry basket', 'shopping basket']:
#         cls_label_shapenet = 'basket'
#     elif cls_label_scannetpp in ['washing machine']:
#         cls_label_shapenet = 'washer'
#     elif cls_label_scannetpp in ['dishwasher']:
#         cls_label_shapenet = 'dishwasher'
#     elif cls_label_scannetpp in ['laptop']:
#         cls_label_shapenet = 'laptop'
#     elif cls_label_scannetpp in ['nightstand', 'night stand']:
#         cls_label_shapenet = 'cabinet'
#     elif cls_label_scannetpp in ['dresser']:
#         cls_label_shapenet = 'cabinet'
#     elif cls_label_scannetpp in ['bicycle', 'bike']:
#         cls_label_shapenet = 'motorbike'
#     elif cls_label_scannetpp in ['guitar', 'guitar bag', 'guitar case', 'electric guitar']:
#         cls_label_shapenet = 'guitar'
#     elif cls_label_scannetpp in ['clock', 'wall clock', 'table clock']:
#         cls_label_shapenet = 'clock'
#     elif cls_label_scannetpp in ['piano']:
#         cls_label_shapenet = 'piano'
#     elif cls_label_scannetpp in ['bowl']:
#         cls_label_shapenet = 'bowl'
#
#     return cls_label_shapenet
# shapenet_category_dict = {'airplane': '02691156', 'trash bin': '02747177', 'bag': '02773838', 'basket': '02801938',
#                           'bathtub': '02808440', 'bed': '02818832', 'bench': '02828884', 'birdhouse': '02843684',
#                           'bookshelf': '02871439', 'bottle': '02876657', 'bowl': '02880940', 'bus': '02924116',
#                           'cabinet': '02933112', 'camera': '02942699', 'can': '02946921', 'cap': '02954340',
#                           'car': '02958343', 'cellphone': '02992529', 'chair': '03001627', 'clock': '03046257',
#                           'keyboard': '03085013', 'dishwasher': '03207941', 'display': '03211117',
#                           'earphone': '03261776', 'faucet': '03325088', 'file cabinet': '03337140', 'guitar': '03467517',
#                           'helmet': '03513137', 'jar': '03593526', 'knife': '03624134', 'lamp': '03636649',
#                           'laptop': '03642806', 'loudspeaker': '03691459', 'mailbox': '03710193',
#                           'microphone': '03759954', 'microwaves': '03761084', 'motorbike': '03790512',
#                           'mug': '03797390', 'piano': '03928116', 'pillow': '03938244', 'pistol': '03948459',
#                           'flowerpot': '03991062', 'printer': '04004475', 'remote': '04074963', 'rifle': '04090263',
#                           'rocket': '04099429', 'skateboard': '04225987', 'sofa': '04256520', 'stove': '04330267',
#                           'table': '04379243', 'telephone': '04401088', 'tower': '04460130', 'train': '04468005',
#                           'watercraft': '04530566', 'washer': '04554684','desk':'03179701', 'dresser':'03237340',
#                           'pillow':'03938244', 'bed cabinet': '20000008'}
#
# def convert_to_cad_format(scene_mesh, object_classes, object_masks, output_dir):
#     scene_vertices = scene_mesh.vertices
#     scene_name = "123"
#
#     obj_3d_list = []
#     inst_label_map = np.zeros((scene_vertices.shape[0], 1))
#     for inst_id, (obj_pred_label, obj_mask) in enumerate(zip(object_classes, object_masks)):
#         # 从 ScanNet200映射到原本的Scannet label id
#         # 具体的表格可以参考 `label_mapping_occ11.csv`
#         obj_class_name = CLASS_LABELS_200[obj_pred_label]
#         obj_class_id = VALID_CLASS_IDS_200[obj_pred_label]
#
#         # 判断这个object是否属于shapnet
#         shapenet_class_name = parse_cls_label[obj_class_name]
#         if shapenet_class_name is None:
#             continue
#         shapnet_catid_cad = shapenet_class_name[shapenet_class_name]
#
#         object_vertices = np.asarray(scene_vertices)[obj_mask]
#
#         # 计算每个obj的obb （OrientedBoundingBox）
#         obb = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(object_vertices))
#
#
#
#
#
#
#         # 可视化
#         box_final = get_corners_of_bb3d_no_index(obb.R.T, obb.extent / 2, obb.center)
#         lineSets_selected = drawOpen3dCylLines([box_final], [0, 1, 0])
#
#
#
#     ##
#
#     view_sel_path = os.path.join(prepro_out_path, scene_name + 'view_selection.pkl')
#     print('Start view selection')
#     if os.path.exists(view_sel_path):
#         pkl_file = open(view_sel_path, 'rb')
#         view_selection_dict = pickle.load(pkl_file)
#         pkl_file.close()
#     else:
#         view_selection_dict = view_selection_new_pose(scene_name, tmesh, frame_id_pose_dict, new_intrinsics,
#                                                       dist_params, img_scale, max_views, silhouette_thres,
#                                                       inst_label_list)
#     if view_selection_dict is None:
#         continue
#
#     obj_instance = ObjectAnnotation(
#         object_id,
#         shapenet_cls_label,
#         scannet_category_label=None,
#         view_params=view_params,
#         transform3d=transform3d,
#         transform_dict=transform_dict,
#         catid_cad=shapnet_catid_cad,
#         scan2cad_annotation_dict=annotation
#     )
#
#
#
#     scene_obj = ScanNetAnnotation(
#         scene_name,
#         obj_3d_list,
#         inst_label_map,
#         scene_type=None
#     )


# def save_instance_segmentation(output_dir)