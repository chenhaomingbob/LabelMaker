import os
import argparse
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Union

from os.path import abspath, dirname, join

import cv2
import gin
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.engine import DefaultTrainer
from nltk.corpus import wordnet as wn
from tqdm import tqdm
from PIL import Image

from labelmaker.label_data import get_ade150, get_replica, get_wordnet, get_occ11

sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', '3rdparty', 'SAN'))
from san import add_san_config

logging.basicConfig(level="INFO")
log = logging.getLogger('SAN Segmentation')


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def load_san(
        device: Union[str, torch.device],
):
    """

    Args:
        device:
        custom_templates:

    Returns:
        Return san model
    """
    config_file = str(
        abspath(
            join(__file__, '../..', '3rdparty', 'SAN', 'configs',
                 'san_clip_vit_large_res4_coco.yaml')
        )
    )

    model_path = abspath(
        join(__file__, '../..', 'checkpoints', 'san_vit_large_14.pth')
    )

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_san_config(cfg)

    cfg.merge_from_file(config_file)
    cfg.MODEL.DEVICE = device
    cfg.freeze()

    model = DefaultTrainer.build_model(cfg)

    log.info('[SAN] Loading model from: {}'.format(model_path))
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        model_path
    )
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    return model


def process_image(
        model,
        img_path,
        class_names,
        threshold=0.7,
        flip=False,
):
    # preprocess
    img = Image.open(img_path).convert("RGB")
    if flip:
        img = Image.fromarray(np.array(img)[:, ::-1])

    w, h = img.size
    # if w < h:
    #     img = img.resize((640, int(h * 640 / w)))
    # else:
    #     img = img.resize((int(w * 640 / h), 640))
    img = torch.from_numpy(np.asarray(img)).float()
    img = img.permute(2, 0, 1)

    # model inference
    with torch.no_grad():
        predictions = model(
            [
                {
                    "image": img,
                    "height": h,
                    "width": w,
                    "vocabulary": class_names,
                }
            ]
        )[0]

    # torch.
    # torch.softmax(predictions['sem_seg'], dim=0)
    # product, pred = torch.max(torch.softmax(predictions['sem_seg'],dim=0), dim=0)
    product, pred = torch.max(predictions['sem_seg'], dim=0)


    # 0 is empty
    # ceiling 0+1 =1
    pred = pred + 1

    # pred[pred >= len(class_names)] = len(class_names)

    # map unknown region to 0
    # pred[product < threshold] = 0

    pred = pred.to('cpu').numpy().astype(int)

    if flip:
        pred = pred[:, ::-1]

    # pred = pred + 1

    return pred


def get_vocabulary(classes):
    if classes == 'occ11':
        classes_data = get_occ11()

        # 按照 'id' 排序
        sorted_data = sorted(classes_data, key=lambda x: x['id'])

        # 提取 'name' 字段
        vocabulary = [item['name'] for item in sorted_data]

        # 移除empty
        vocabulary = vocabulary[1:]

    else:
        raise ValueError(f'Unknown class set {classes}')

    return vocabulary


@gin.configurable
def run(
        scene_dir: Union[str, Path],
        output_folder: Union[str, Path],
        device: Union[str, torch.device] = 'cuda:0',
        # changing this to cuda default as all of us have it available. Otherwise, it will fail on machines without cuda
        classes='occ11',  # for open vocabulary method
        flip: bool = False,
):
    # convert str to Path object
    scene_dir = Path(scene_dir)
    output_folder = Path(output_folder)

    # check if scene_dir exists
    assert scene_dir.exists() and scene_dir.is_dir()

    input_color_dir = scene_dir / 'color'
    assert input_color_dir.exists() and input_color_dir.is_dir()

    output_dir = scene_dir / output_folder
    output_dir = Path(str(output_dir) + '_flip') if flip else output_dir

    # only for open vocabulary method
    if classes != 'occ11':
        output_dir.replace('occ11', classes)

    # check if output directory exists
    shutil.rmtree(output_dir, ignore_errors=True)  # remove output_dir if it exists
    os.makedirs(str(output_dir), exist_ok=False)

    input_files = input_color_dir.glob('*')
    input_files = sorted(input_files, key=lambda x: int(x.stem.split('_')[-1]))

    log.info(f'[san] using {classes} classes')
    log.info(f'[san] inference in {str(input_color_dir)}')

    # templates, class_names = get_templates(classes)
    # id_map = get_id_map(classes)

    class_names = get_vocabulary(classes)

    log.info('[san] loading model')
    model = load_san(device=device)

    log.info('[san] inference')

    for file in tqdm(input_files):
        result = process_image(model, file, class_names, flip=flip)

        cv2.imwrite(
            str(output_dir / f'{file.stem}.png'),
            result.astype(np.uint16),
        )


def arg_parser():
    parser = argparse.ArgumentParser(description='SAN Segmentation')
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
        default='intermediate/occ11_san_1',
        help=
        'Name of output directory in the workspace directory intermediate. Has to follow the pattern $labelspace_$model_$version',
    )
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument(
        '--flip',
        action="store_true",
        help='Flip the input image, this is part of test time augmentation.',
    )
    parser.add_argument('--config', help='Name of config file')
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parser()
    if args.config is not None:
        gin.parse_config_file(args.config)

    setup_seeds(seed=args.seed)
    run(scene_dir=args.workspace, output_folder=args.output, flip=args.flip)
