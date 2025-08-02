import os
from pathlib import Path
from typing import Set, List


def find_folders_depth_limited(start_dir: str = ".", filename: str = "pc_voxel_down.ply",
                               foldername: str = "intermediate") -> Set[Path]:
    """
    在指定目录中搜索，深度限制为两层，找到那些在其子目录里包含特定文件的文件夹。

    Args:
        start_dir (str): 搜索的起始目录。
        filename (str): 要查找的目标文件名。

    Returns:
        Set[Path]: 一个包含所有符合条件的父文件夹路径的集合。
    """
    start_path = Path(start_dir)
    # 使用集合 (set) 来自动处理重复的文件夹路径
    found_folders_by_filename = set()
    found_folders_by_foldername = set()

    print(f"正在从 '{start_path.resolve()}' 开始搜索（深度限制为2层）...")

    # --- 情况 1: 检查文件是否在第一层子目录中 ---
    # 模式: */文件名, 例如 './一级目录/pc_voxel_down.ply'
    # glob('*/filename') 会匹配到文件本身
    for file_path in start_path.glob(f"*/{filename}"):
        if file_path.is_file():
            # 文件在'一级目录'下，所以要返回的父文件夹是当前目录 (start_path)
            parent_of_start_path = file_path.resolve().parent
            found_folders_by_filename.add(parent_of_start_path)

    for folder_path in start_path.glob(f"*/{foldername}"):
        if folder_path.is_dir():
            # 文件在'一级目录'下，所以要返回的父文件夹是当前目录 (start_path)
            parent_of_start_path = folder_path.resolve().parent
            found_folders_by_foldername.add(parent_of_start_path)

    return found_folders_by_filename, found_folders_by_foldername


if __name__ == "__main__":
    # 定义要搜索的目标文件名
    TARGET_FILENAME = "pc_voxel_down.ply"
    TARGET_FOLDERNAME = "intermediate"

    # 执行搜索
    found_folders_by_file_name, found_folders_by_folder_name = find_folders_depth_limited(
        start_dir="/data1/chm/datasets/arkitscenes/LabelMaker/Training",
        filename=TARGET_FILENAME,
        foldername=TARGET_FOLDERNAME
    )

    print(f"包含 {TARGET_FILENAME}的文件夹个数", len(found_folders_by_file_name))
    print(f"包含 {TARGET_FOLDERNAME}的文件夹个数", len(found_folders_by_folder_name))

    if found_folders_by_file_name:
        print("\n检索完成。找到以下符合条件的文件夹：")
        # 对结果进行排序，以便输出顺序一致
        sorted_folders: List[Path] = sorted(list(found_folders_by_file_name))

        sorted_folders = sorted_folders[:600]
        # for folder in enumerate(sorted_folders):
        # os.path.normpath 可以清理路径表示 (例如将 './folder' 转换为 'folder', './' 转换为 '.')
        # print(f"ln -s {os.path.normpath(folder)}")
        # 将所有 ln -s 命令合并为一行
        commands = ' && '.join(f'ln -s {os.path.normpath(folder)}' for folder in sorted_folders)
        print(commands)
    else:
        print(f"\n检索完成。在两层深度内，未找到任何子目录下包含 '{TARGET_FILENAME}' 的文件夹。")

    if found_folders_by_folder_name:
        print("\n检索完成。找到以下符合条件的文件夹：")
        # 对结果进行排序，以便输出顺序一致
        sorted_folders: List[Path] = sorted(list(found_folders_by_file_name))
        for folder in sorted_folders:
            # os.path.normpath 可以清理路径表示 (例如将 './folder' 转换为 'folder', './' 转换为 '.')
            print(f"ln -s {os.path.normpath(folder)}")
        # 将所有 ln -s 命令合并为一行
        commands = ' && '.join(f'ln -s {os.path.normpath(folder)}' for folder in sorted_folders)
        print(commands)
    else:
        print(f"\n检索完成。在两层深度内，未找到任何子目录下包含 '{TARGET_FILENAME}' 的文件夹。")
