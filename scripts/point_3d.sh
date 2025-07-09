#!/bin/bash

start_time=$SECONDS
echo "脚本开始运行..."
# -----------------------------------------------------------------------------
# process_workspaces.sh
#
# 该脚本用于对一个指定父目录下的所有第一层子目录运行一系列语义标注模型。
#
# 使用方法:
#   chmod +x process_workspaces.sh
#   ./process_workspaces.sh /path/to/parent_directory
#
# 示例:
#   ./process_workspaces.sh /data1/chm/datasets/arkitscenes/LabelMaker/Training
#   (脚本会自动处理 Training 目录下的 42444913, ws1, ws2 等子目录)
# -----------------------------------------------------------------------------

# 检查是否提供了且仅提供了一个父目录路径
if [ $# -ne 1 ]; then
    echo "错误: 请提供一个父目录路径作为参数。"
    echo "用法: $0 /path/to/parent_directory"
    exit 1
fi

PARENT_DIR="$1"

# 检查父目录是否存在
if [ ! -d "$PARENT_DIR" ]; then
    echo "错误: 父目录不存在: $PARENT_DIR"
    exit 1
fi

echo "=========================================================="
echo ">>> 开始扫描父目录: $PARENT_DIR"
echo "=========================================================="


# --- 统计子目录总数 ---
# 使用 find 命令安全地统计第一层子目录的数量
total_dirs=$(find "$PARENT_DIR" -maxdepth 1 -mindepth 1 | wc -l)

if [ "$total_dirs" -eq 0 ]; then
    echo "!!! 警告: 在 '$PARENT_DIR' 中没有找到任何子目录。"
    exit 0
fi

echo "[INFO] 统计完成：共发现 ${total_dirs} 个子目录需要处理。"
echo "" # 增加空行，方便阅读

# 遍历父目录下的所有第一层子目录
# "$PARENT_DIR"/*/ 这个通配符会匹配所有子目录
for workspace in "$PARENT_DIR"/*/; do
    workspace_start_time=$SECONDS
    # 如果父目录中没有子目录，通配符可能不会展开，所以这里加一个检查
    if [ ! -d "$workspace" ]; then
        echo "!!! 警告: 在 '$PARENT_DIR' 中没有找到任何子目录，或 '$workspace' 不是一个目录。"
        continue # 跳过
    fi



    # 检查当前 workspace 是否存在 intermediate 文件夹
    intermediate_dir="$workspace/intermediate"
    if [ ! -d "$intermediate_dir" ]; then
        echo "[WARNING] 未找到 'intermediate' 文件夹，跳过当前工作区: $workspace"
        echo ""
        continue # 如果文件夹不存在，则跳过本次循环，处理下一个工作区
    fi

    # --- 修改：增加处理进度 ---
    ((processed_count++))
    workspace_start_time=$SECONDS

        # 检查目标文件是否存在于工作区内
    if [ -f "$workspace/point_lifted_mesh.ply" ]; then
        echo "[INFO] 检测到 'point_lifted_mesh.ply'，跳过已处理的工作区: $workspace"
        echo ""
        continue # 如果文件存在，则跳过本次循环，处理下一个工作区
    fi

    echo "=========================================================="
    echo ">>> 开始处理第 ${processed_count} / ${total_dirs} 个工作区: $workspace"
    echo "=========================================================="

    echo "[INFO] 正在运行共识投票..."
    python labelmaker/consensus_occ11.py --workspace "$workspace"
    echo "[SUCCESS] 共识投票完成。"

    echo "[INFO] 正在进行3D语义标注..."
    python -m labelmaker.lifting_3d.lifting_points --workspace "$workspace"
    echo "[SUCCESS] 3D语义标注提升完成。"

    echo "[INFO] 正在运行mesh转occupancy..."
    python labelmaker/occupancy/mesh2occupancy.py  --workspace "$workspace"
    echo "[SUCCESS] 转换完成。"


    workspace_end_time=$SECONDS
    workspace_duration=$((workspace_end_time - workspace_start_time))

    echo "----------------------------------------------------------"
    echo ">>> 工作区处理完成: $workspace"
    printf ">>> 该工作区耗时: %d 分 %d 秒\n" $((workspace_duration / 60)) $((workspace_duration % 60))
    echo "----------------------------------------------------------"
    echo ""
done

end_time=$SECONDS
duration=$((end_time - start_time))

echo "=========================================================="
echo "所有 ${total_dirs} 个子目录均已处理完毕。"
printf "脚本总耗时: %d 分 %d 秒\n" $((duration / 60)) $((duration % 60))
echo "=========================================================="