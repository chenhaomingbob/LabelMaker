import os
import numpy as np
import glob
import pandas as pd
from collections import Counter
from tqdm import tqdm


def calculate_sky_direction_from_pose(R):
    """
    根据相机位姿（外参旋转矩阵）计算图像中的天空方向。
    （此函数与之前版本相同）
    """
    if R.shape == (4, 4):
        R = R[:3, :3]
    up_world = np.array([[0], [0], [1]])
    up_cam = R[:, 2]
    x_c, y_c = up_cam[0], up_cam[1]
    angle_current_rad = np.arctan2(y_c, x_c)
    angle_target_rad = -np.pi / 2
    rotation_rad = angle_target_rad - angle_current_rad
    rotation_deg_quantized = round(np.rad2deg(rotation_rad) / 90) * 90
    if rotation_deg_quantized < 0:
        rotation_deg_quantized += 360
    rotation_deg_quantized = int(rotation_deg_quantized % 360)

    if rotation_deg_quantized == 270:
        return 'Right'
    elif rotation_deg_quantized == 90:
        return 'Left'
    elif rotation_deg_quantized == 180:
        return 'Down'
    else:
        return 'Up'


def calculate_representative_directions(dataset_root):
    """
    遍历数据集，为每个场景（视频）计算出最具代表性的天空方向。

    返回:
    dict: 一个字典，键是 scene_id，值是该场景最具代表性的天空方向。
    """
    scene_paths = sorted([d for d in glob.glob(os.path.join(dataset_root, '*')) if os.path.isdir(d)])

    # 用于存储每个场景的代表方向
    representative_dirs = {}

    print("正在计算每个视频的代表性天空方向...")
    for scene_path in tqdm(scene_paths, desc="处理场景中"):
        scene_id = os.path.basename(scene_path)
        # if scene_id != '41126944':
        #     continue
        # if scene_id != '48018514':
        #     continue
        pose_dir = os.path.join(scene_path, 'pose')

        if not os.path.isdir(pose_dir):
            continue

        directions_for_this_scene = []
        pose_files = sorted(glob.glob(os.path.join(pose_dir, '*.txt')))

        pose_files = pose_files[::5]
        for idx, pose_path in enumerate(pose_files):
            try:
                world_to_camera_pose = np.linalg.inv(np.loadtxt(pose_path))
                sky_direction = calculate_sky_direction_from_pose(world_to_camera_pose)
                directions_for_this_scene.append(sky_direction)
                if idx >= 100:
                    break
            except Exception:
                # 忽略处理失败的帧
                continue

        # 如果成功处理了该场景的帧，则计算众数
        if directions_for_this_scene:
            # 使用Counter找到出现次数最多的方向
            most_common_direction = Counter(directions_for_this_scene).most_common(1)[0][0]
            representative_dirs[scene_id] = most_common_direction

    return representative_dirs


def main():
    """主执行函数"""

    # --- 请修改以下路径 ---
    # 1. 数据集根目录
    DATASET_ROOT_PATH = "/data1/chm/datasets/arkitscenes/LabelMaker/Training/"

    # 2. 您已有的CSV文件路径
    EXISTING_CSV_PATH = "/data1/chm/datasets/arkitscenes/LabelMaker/metadata.csv"  # <--- 请务必修改为您的CSV文件路径

    # 3. 更新后要保存的新文件名
    OUTPUT_CSV_PATH = "/data1/chm/datasets/arkitscenes/LabelMaker/metadata_updated.csv"
    # --- 修改结束 ---

    # 步骤 1: 计算每个视频的代表性天空方向
    representative_sky_directions = calculate_representative_directions(DATASET_ROOT_PATH)
    if not representative_sky_directions:
        print("未能计算出任何视频的天空方向，程序终止。")
        return

    print(f"\n步骤 1 完成：已为 {len(representative_sky_directions)} 个视频计算出代表方向。")

    # 步骤 2: 读取现有CSV文件并合并数据
    print(f"步骤 2: 正在读取现有文件 '{EXISTING_CSV_PATH}'...")
    try:
        df = pd.read_csv(EXISTING_CSV_PATH)
    except FileNotFoundError:
        print(f"错误：找不到文件 '{EXISTING_CSV_PATH}'。请检查路径是否正确。")
        return

    # 将'video_id'列转为字符串类型，以确保与字典的键类型匹配
    df['video_id'] = df['video_id'].astype(str)

    # 使用.map()方法，根据video_id匹配字典中的值，创建新列
    print("步骤 3: 正在添加新列 'cal_sky_direction'...")
    df['cal_sky_direction'] = df['video_id'].map(representative_sky_directions)

    # 对于在数据集中没有找到对应video_id的行，可以用一个默认值填充
    df['cal_sky_direction'].fillna('NA', inplace=True)
    # df['visit_id'].fillna('NA', inplace=True)

    # 步骤 3: 保存到新的CSV文件
    df.to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"\n处理完成！已将更新后的数据保存到: '{OUTPUT_CSV_PATH}'")
    print("\n新文件内容预览:")
    print(df[['video_id', 'sky_direction', 'cal_sky_direction']].head())


# --- 主程序入口 ---
if __name__ == '__main__':
    main()
