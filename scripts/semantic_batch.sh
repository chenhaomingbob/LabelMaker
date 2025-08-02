#!/bin/bash

# ==============================================================================
# 多进程多GPU并行处理脚本 (后台作业控制版本)
# - 修复了xargs导致GPU任务分配不均的问题
# - 为每个GPU创建独立的任务队列
# ==============================================================================

# --- 1. 用户配置区 ---
if [ $# -ne 1 ]; then
    echo "错误: 请提供一个父目录路径作为参数。"
    exit 1
fi
PARENT_DIR="$1"
if [ ! -d "$PARENT_DIR" ]; then
    echo "错误: 父目录不存在: $PARENT_DIR"
    exit 1
fi

# --- 2. GPU和进程数配置 ---
NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [ "$NUM_GPUS" -eq 0 ]; then
    echo "错误: 未检测到任何NVIDIA GPU，或者 nvidia-smi 命令执行失败。"
    exit 1
fi
echo "[INFO] 检测到 ${NUM_GPUS} 块GPU。"

# --- 统计功能设置 ---
STATUS_DIR="/tmp/semantic_status_$$"
mkdir -p "$STATUS_DIR"
trap 'echo "正在清理临时文件..."; rm -rf "$STATUS_DIR"; echo "清理完毕。"' EXIT

# --- 3. 核心处理函数 (仅接收工作区路径) ---
# GPU ID 将通过环境变量传入
process_single_workspace() {
    local workspace="$1"
    # 函数内部不再接收gpu_id，直接从环境变量读取
    local gpu_id=$CUDA_VISIBLE_DEVICES
    local ws_start_time=$SECONDS

    # 检查 'intermediate' 文件夹
    if [ -d "$workspace/intermediate/occ11_san_1" ]; then
        echo "[GPU ${gpu_id}] 检测到 'intermediate/occ11_san_1_flip' 文件夹，跳过: $workspace"
        touch "$STATUS_DIR/$(basename "$workspace").success"
        return 0
    fi
    # 检查 'point_lifted_mesh.ply' 文件
    if [ -f "$workspace/point_lifted_mesh.ply" ]; then
        echo "[GPU ${gpu_id}] 检测到 'point_lifted_mesh.ply'，跳过: $workspace"
        touch "$STATUS_DIR/$(basename "$workspace").success"
        return 0
    fi

    echo "[GPU ${gpu_id}] 开始处理工作区: $workspace"
    # --- 运行所有模型脚本 (这部分不变) ---
    python models/internimage.py --workspace "$workspace" && \
    python models/internimage.py --flip --workspace "$workspace" && \
    python models/ovseg.py --workspace "$workspace" && \
    python models/ovseg.py --flip --workspace "$workspace" && \
    python models/grounded_sam.py --workspace "$workspace" && \
    python models/grounded_sam.py --flip --workspace "$workspace" && \
    python models/omnidata_depth.py --workspace "$workspace" && \
    python models/hha_depth.py --workspace "$workspace" && \
    python models/cmx.py --workspace "$workspace" && \
    python models/mask3d_inst.py --workspace "$workspace" --seed 42 && \
    python models/mask3d_inst.py --workspace "$workspace" --seed 43 --output intermediate/scannet200_mask3d_2 && \
    python models/sanseg.py --workspace "$workspace" && \
    python models/sanseg.py --flip --workspace "$workspace"

    local exit_code=$?
    local ws_duration=$((SECONDS - ws_start_time))
    if [ $exit_code -eq 0 ]; then
        echo "[GPU ${gpu_id}] [SUCCESS] 成功处理: $workspace"
        touch "$STATUS_DIR/$(basename "$workspace").success"
    else
        echo "[GPU ${gpu_id}] [ERROR] 处理失败: $workspace"
    fi
    printf "[GPU ${gpu_id}] --- 工作区 '%s' 处理耗时: %d 分 %d 秒 ---\n" "$(basename "$workspace")" $((ws_duration / 60)) $((ws_duration % 60))
}

export -f process_single_workspace
export STATUS_DIR

PROCS_PER_GPU=2  # 每个GPU并行处理的最大任务数
# --- 4. 任务分配与并行执行 (新逻辑) ---
echo "=========================================================="
echo ">>> 开始扫描并分配任务到各个GPU..."
echo ">>> 每个GPU将并行运行最多 ${PROCS_PER_GPU} 个任务。"
echo "=========================================================="
overall_start_time=$SECONDS

# 获取所有待处理的工作区列表
mapfile -t all_workspaces < <(find "$PARENT_DIR" -maxdepth 1 -mindepth 1 -type d,l)
total_workspaces=${#all_workspaces[@]}
if [ "$total_workspaces" -eq 0 ]; then
    echo "警告: 在 '$PARENT_DIR' 中没有找到任何子目录。"
    exit 0
fi
echo "[INFO] 共发现 ${total_workspaces} 个工作区需要检查和处理。"

## 为每个GPU启动一个独立的“任务处理器”
#for (( gpu_id=0; gpu_id<NUM_GPUS; gpu_id++ )); do
#    # 这个括号内的代码块会在一个子shell中异步执行
#    (
#        # 设置当前处理器可见的GPU
#        export CUDA_VISIBLE_DEVICES=$gpu_id
#        echo "[INFO] GPU ${gpu_id} 任务处理器已启动。"
#
#        # 遍历所有工作区，只处理分配给自己的任务
#        for (( i=gpu_id; i<total_workspaces; i+=NUM_GPUS )); do
#            workspace_path="${all_workspaces[i]}"
#            # 调用核心处理函数
#            process_single_workspace "$workspace_path"
#        done
#    ) & # & 符号表示将这个子shell放到后台运行
#done


# 为每个GPU启动一个独立的“任务处理器”
for (( gpu_id=0; gpu_id<NUM_GPUS; gpu_id++ )); do
    # 这个括号内的代码块会在一个子shell中异步执行
    (
        # Set the visible GPU for this specific subshell
        export CUDA_VISIBLE_DEVICES=$gpu_id
        echo "[INFO] GPU ${gpu_id} 任务处理器已启动。"

        # Iterate through all workspaces assigned to this GPU
        for (( i=gpu_id; i<total_workspaces; i+=NUM_GPUS )); do
            # *** JOB CONTROL LOGIC START ***
            # Check the number of active background jobs in this subshell
            # and wait if the limit is reached.
            while [[ $(jobs -p | wc -l) -ge $PROCS_PER_GPU ]]; do
                # Wait for any single job to finish before checking again
                wait -n
            done
            # *** JOB CONTROL LOGIC END ***

            workspace_path="${all_workspaces[i]}"

            # Call the core processing function IN THE BACKGROUND
            process_single_workspace "$workspace_path" &
        done

        # After the loop, wait for all remaining background jobs for this GPU to complete
        echo "[INFO] GPU ${gpu_id}: 所有任务已启动，正在等待处理完成..."
        wait
        echo "[INFO] GPU ${gpu_id}: 所有任务处理完毕。"

    ) & # & places this entire per-GPU handler into the background
done

# 等待所有后台的GPU处理器完成任务
echo ">>> 所有任务已分配，等待后台进程执行完毕..."
wait

# --- 5. 最终统计和报告 (这部分不变) ---
overall_duration=$((SECONDS - overall_start_time))
success_count=$(find "$STATUS_DIR" -type f -name "*.success" | wc -l)
fail_count=$((total_workspaces - success_count))

echo "=========================================================="
echo "所有工作区并行处理完毕！"
echo "--- 处理结果摘要 ---"
echo "总计工作区: ${total_workspaces}"
echo -e "\033[0;32m成功 (或跳过): ${success_count}\033[0m"
echo -e "\033[0;31m失败: ${fail_count}\033[0m"
printf "脚本总耗时: %d 分 %d 秒\n" $((overall_duration / 60)) $((overall_duration % 60))
echo "=========================================================="