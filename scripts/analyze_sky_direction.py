import os
import pandas as pd

# --- 配置 ---
# 1. 数据集文件夹的路径
# 请根据您的环境修改此路径
dataset_dir = '/data1/chm/datasets/arkitscenes/LabelMaker/Training'
# dataset_dir = '/data1/chm/datasets/arkitscenes/LabelMaker/mini_data_v3'

# 2. 包含 video_id 和 cal_sky_direction 的CSV文件路径
# 请将 'path/to/your/spreadsheet.csv' 替换为您的CSV文件的实际路径
csv_file_path = '/data1/chm/datasets/arkitscenes/LabelMaker/metadata_updated.csv'


# --- 脚本开始 ---

def analyze_and_list_video_ids(directory, csv_path):
    """
    根据目录中的 video_id，统计并列出 CSV 文件中
    每个 cal_sky_direction 对应的 video_id 列表。

    参数:
    - directory (str): 包含以 video_id 命名的子文件夹的路径。
    - csv_path (str): CSV 文件的路径。
    """
    # 检查数据集目录是否存在
    if not os.path.isdir(directory):
        print(f"错误: 目录不存在 -> '{directory}'")
        return

    # 检查CSV文件是否存在
    if not os.path.exists(csv_path):
        print(f"错误: CSV文件不存在 -> '{csv_path}'")
        print("请确保在脚本中设置了正确的 'csv_file_path'。")
        return

    print(f"正在从 '{directory}' 读取文件夹列表...")

    # 步骤1: 获取目录下所有文件夹名称 (video_id)
    folder_video_ids = set()
    for item in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, item)):
            try:
                folder_video_ids.add(int(item))
            except ValueError:
                pass

    if not folder_video_ids:
        print(f"错误: 在 '{directory}' 中没有找到有效的 video_id 文件夹。")
        return

    print(f"成功找到 {len(folder_video_ids)} 个 video_id 文件夹。")
    print("-" * 40)

    print(f"正在读取并分析CSV文件: '{csv_path}'...")

    # 步骤2: 读取CSV文件
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"错误: 读取CSV文件时出错: {e}")
        return

    required_columns = ['video_id', 'cal_sky_direction']
    if not all(col in df.columns for col in required_columns):
        print(f"错误: CSV文件必须包含以下列: {required_columns}")
        return

    # 步骤3: 筛选出 video_id 在文件夹列表中的行
    filtered_df = df[df['video_id'].isin(folder_video_ids)].copy()

    # 将 'cal_sky_direction' 中的 NaN 值替换为字符串 'NA'，以便分组
    filtered_df['cal_sky_direction'].fillna('NA', inplace=True)

    if filtered_df.empty:
        print("错误: 在CSV文件中没有找到任何与文件夹匹配的 video_id。")
        return

    print(f"在CSV中找到了 {len(filtered_df)} 条与文件夹匹配的记录。")
    print("-" * 40)

    # 步骤4: 按 'cal_sky_direction' 分组
    grouped_by_direction = filtered_df.groupby('cal_sky_direction')

    # --- 输出结果 ---
    print("按 cal_sky_direction 分类的 Video ID 列表:")

    # 遍历每个分组并打印信息
    for direction, group_df in grouped_by_direction:
        # 获取当前分组的所有 video_id
        video_ids_list = group_df['video_id'].tolist()

        print(f"\n方向 (cal_sky_direction): {direction}")
        print(f"  数量: {len(video_ids_list)}")
        print(f"  对应的 Video IDs: {video_ids_list}")


# --- 运行主函数 ---
if __name__ == "__main__":
    analyze_and_list_video_ids(dataset_dir, csv_file_path)
