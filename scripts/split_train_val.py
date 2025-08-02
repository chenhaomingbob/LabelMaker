import os
import random


def create_split_files_by_scenes(data_dir, split_ratio=0.8,by_scene=False):
    """
    扫描指定目录下的场景文件夹，收集所有图像帧（color/*.jpg），按场景划分训练集和验证集，
    并生成对应的 train.txt 和 val.txt 文件，每行格式为 "scene_id/frame_id"。

    参数:
    data_dir (str): 包含所有场景文件夹的数据集根目录。
                    例如: '/data1/chm/datasets/arkitscenes/LabelMaker/mini_data'
    split_ratio (float): 训练集所占比例，默认为0.8。
    """
    print(f"正在扫描目录: {data_dir}")

    # 1. 检查目录是否存在
    if not os.path.isdir(data_dir):
        print(f"错误: 目录 '{data_dir}' 不存在。请检查路径是否正确。")
        return

    # 2. 收集所有场景 ID，并过滤掉缺少 color 或 preprocessed_voxels 目录的场景
    all_scenes = []
    try:
        for scene_id in os.listdir(data_dir):
            scene_path = os.path.join(data_dir, scene_id)
            if not os.path.isdir(scene_path):
                continue

            color_dir = os.path.join(scene_path, 'color')
            voxels_dir = os.path.join(scene_path, 'preprocessed_voxels')

            if not os.path.exists(color_dir) or not os.path.exists(voxels_dir):
                print(f"警告: 场景 '{scene_id}' 缺少 color 或 preprocessed_voxels 目录，跳过该场景。")
                continue

            all_scenes.append(scene_id)

    except Exception as e:
        print(f"错误: 扫描过程中发生异常。原因: {e}")
        return

    if not all_scenes:
        print("错误: 在指定目录中没有找到任何有效场景。")
        return

    print(f"共找到 {len(all_scenes)} 个有效场景。")

    # 3. 随机打乱场景列表
    random.shuffle(all_scenes)
    print("已将场景列表随机打乱。")

    # 4. 根据比例计算分割点
    split_index = int(len(all_scenes) * split_ratio)

    # 5. 分割场景列表
    train_scenes = all_scenes[:split_index]
    val_scenes = all_scenes[split_index:]

    print(f"划分结果: {len(train_scenes)} 个场景用于训练, {len(val_scenes)} 个场景用于验证。")

    # 6. 收集每个场景下的帧 ID，并写入文件
    train_frames = []
    val_frames = []

    for scene_id in train_scenes:
        color_dir = os.path.join(data_dir, scene_id, 'color')
        voxels_dir = os.path.join(data_dir, scene_id, 'preprocessed_voxels')

        for img_file in os.listdir(color_dir):
            if img_file.lower().endswith('.jpg'):
                frame_id = os.path.splitext(img_file)[0]
                voxel_file_path = os.path.join(voxels_dir, f"{frame_id}.pkl")
                if os.path.exists(voxel_file_path):
                    train_frames.append(f"{scene_id}/{frame_id}")

    for scene_id in val_scenes:
        color_dir = os.path.join(data_dir, scene_id, 'color')
        voxels_dir = os.path.join(data_dir, scene_id, 'preprocessed_voxels')

        for img_file in os.listdir(color_dir):
            if img_file.lower().endswith('.jpg'):
                frame_id = os.path.splitext(img_file)[0]
                voxel_file_path = os.path.join(voxels_dir, f"{frame_id}.pkl")
                if os.path.exists(voxel_file_path):
                    val_frames.append(f"{scene_id}/{frame_id}")

    print(f"训练集中包含 {len(train_frames)} 个帧，验证集包含 {len(val_frames)} 个帧。")

    # 7. 写入到文件
    try:
        train_file_path = os.path.join(data_dir, 'train_by_scene.txt')
        with open(train_file_path, 'w') as f:
            for frame in train_frames:
                f.write(frame + '\n')
        print(f"训练集文件已保存到: {train_file_path}")

        val_file_path = os.path.join(data_dir, 'val_by_scene.txt')
        with open(val_file_path, 'w') as f:
            for frame in val_frames:
                f.write(frame + '\n')
        print(f"验证集文件已保存到: {val_file_path}")

    except IOError as e:
        print(f"错误: 写入文件失败。原因: {e}")

    print("\n脚本执行完毕！")



def create_split_files_by_frames(data_dir, split_ratio=0.8):
    """
    扫描指定目录下的场景文件夹，收集所有图像帧（color/*.jpg），随机划分为训练集和验证集，
    并生成对应的 train.txt 和 val.txt 文件，每行格式为 "scene_id/frame_id"。

    参数:
    data_dir (str): 包含所有场景文件夹的数据集根目录。
                    例如: '/data1/chm/datasets/arkitscenes/LabelMaker/mini_data'
    split_ratio (float): 训练集所占比例，默认为0.8。
    """
    print(f"正在扫描目录: {data_dir}")

    # 1. 检查目录是否存在
    if not os.path.isdir(data_dir):
        print(f"错误: 目录 '{data_dir}' 不存在。请检查路径是否正确。")
        return

    # 2. 收集所有帧 ID：格式为 "scene_id/frame_id"
    all_frames = []
    skipped_frames_count = 0
    try:
        # 遍历所有场景文件夹
        for scene_id in os.listdir(data_dir):
            scene_path = os.path.join(data_dir, scene_id)
            if not os.path.isdir(scene_path):
                continue

            color_dir = os.path.join(scene_path, 'color')
            if not os.path.exists(color_dir):
                print(f"警告: 场景 '{scene_id}' 中缺少 color 目录，跳过该场景。")
                continue

            voxels_dir = os.path.join(scene_path, 'preprocessed_voxels')
            if not os.path.exists(voxels_dir):
                print(f"警告: 场景 '{scene_id}' 中缺少 preprocessed_voxels 目录，跳过该场景。")
                continue

            # 收集该场景下所有 .jpg 图像文件名（frame_id）
            for img_file in os.listdir(color_dir):
                if img_file.lower().endswith('.jpg'):
                    frame_id = os.path.splitext(img_file)[0]

                    # --- 新增：检查对应的 .pkl 文件是否存在 ---
                    voxel_file_path = os.path.join(voxels_dir, f"{frame_id}.pkl")
                    if os.path.exists(voxel_file_path):
                        # 只有当 .jpg 和 .pkl 都存在时，才添加该帧
                        all_frames.append(f"{scene_id}/{frame_id}")
                    else:
                        # （可选）如果想知道哪些帧被跳过了，可以取消下面的注释
                        print(f"信息: 帧 '{scene_id}/{frame_id}' 因缺少对应的 .pkl 文件而被跳过。")
                        skipped_frames_count += 1
                    # --- 结束新增 ---
                    # all_frames.append(f"{scene_id}/{frame_id}")

    except Exception as e:
        print(f"错误: 扫描过程中发生异常。原因: {e}")
        return

    if not all_frames:
        print("错误: 在指定目录中没有找到任何图像帧（color/*.jpg）。")
        return

    print(f"共找到 {len(all_frames)} 个图像帧。")

    # 3. 随机打乱帧列表
    random.shuffle(all_frames)
    print("已将图像帧列表随机打乱。")

    # 4. 根据比例计算分割点
    split_index = int(len(all_frames) * split_ratio)

    # 5. 分割列表
    train_frames = all_frames[:split_index]
    val_frames = all_frames[split_index:]

    print(f"划分结果: {len(train_frames)} 个用于训练, {len(val_frames)} 个用于验证。")

    # 6. 将划分结果写入 train.txt 和 val.txt
    try:
        train_file_path = os.path.join(data_dir, 'train_by_frames.txt')
        with open(train_file_path, 'w') as f:
            for frame in train_frames:
                f.write(frame + '\n')
        print(f"训练集文件已保存到: {train_file_path}")

        val_file_path = os.path.join(data_dir, 'val_by_frames.txt')
        with open(val_file_path, 'w') as f:
            for frame in val_frames:
                f.write(frame + '\n')
        print(f"验证集文件已保存到: {val_file_path}")

    except IOError as e:
        print(f"错误: 写入文件失败。原因: {e}")

    print("\n脚本执行完毕！")

if __name__ == '__main__':
    dataset_directory = '/data1/chm/datasets/arkitscenes/LabelMaker/mini_data_v4'
    train_split_ratio = 0.8  # 80% 训练，20% 验证

    create_split_files_by_frames(dataset_directory, train_split_ratio)
    create_split_files_by_scenes(dataset_directory, train_split_ratio)
