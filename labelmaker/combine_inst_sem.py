# 合并语义分割结果+实例分割
from pathlib import Path
import os
import argparse
import logging
from typing import Union
import gin

logging.basicConfig(level="INFO")
log = logging.getLogger('Segmentation Consensus')


@gin.configurable
def run(
        scene_dir: Union[str, Path]
):
    if isinstance(scene_dir, str):
        scene_dir = Path(scene_dir)

    inst_seg_base_dir = scene_dir / 'intermediate' / 'scannet200_mask3d_1'

    instance_segmentation_prediction_file = inst_seg_base_dir / 'predictions.txt'

    with open(instance_segmentation_prediction_file) as f:
        instances = [x.strip().split(' ') for x in f.readlines()]

    # 遍历实例
    for i, inst_items in enumerate(instances):
        pass


def arg_parser():
    parser = argparse.ArgumentParser(description='Run consensus segmentation')
    parser.add_argument(
        '--workspace',
        type=str,
        required=True,
        help='Path to workspace directory. There should be a "color" folder.',
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parser()
    if args.config is not None:
        gin.parse_config_file(args.config)
    run(
        scene_dir=args.workspace,
    )
