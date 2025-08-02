#!/bin/bash

# ==============================================================================
# 脚本功能: 迁移已处理过的原始数据文件夹以节省磁盘空间
# 工作逻辑:
#   1. 检查预处理文件夹 (PREPROCESSED_DIR) 中的场景ID。
#   2. 如果某个场景ID在原始数据文件夹 (RAW_DATA_DIR) 中也存在，
#   3. 则检查 metadata.csv 文件中该ID的 'cal_sky_direction' 属性。
#   4. 仅当属性值为 'Up' 时，才将其从 RAW_DATA_DIR 移动到归档文件夹 (DESTINATION_DIR)。
# 使用：
#   1. 先用模拟一遍
#       ./move_processed_raw_data.sh --dry-run
#   2. 检查无误后，正式启用
#       ./move_processed_raw_data.sh
# ==============================================================================

# --- 1. 用户配置区 (请根据您的实际路径修改) ---

# 原始数据文件夹的路径
RAW_DATA_DIR="/data1/chm/datasets/arkitscenes/raw/Training"

# 预处理后数据文件夹的路径
PREPROCESSED_DIR="/data1/chm/datasets/arkitscenes/LabelMaker/Training"

# 归档文件夹的路径 (移动到这里)
DESTINATION_DIR="/mnt/temp/arkitscenes"

# 【新增】Metadata CSV文件的路径 (假设与脚本在同一目录)
METADATA_CSV="/data1/chm/datasets/arkitscenes/LabelMaker/metadata_updated.csv"


# --- 脚本配置 ---

# 日志文件，记录所有操作
LOG_FILE="move_processed_data_$(date +%Y%m%d_%H%M%S).log"

# --- 颜色定义 (让输出更清晰) ---
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# --- 脚本主逻辑 ---

# 检查是否为预演模式 (dry-run)
DRY_RUN=true
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo -e "${YELLOW}*** 正在以【预演模式】运行，不会实际移动任何文件 ***${NC}\n" | tee -a "$LOG_FILE"
else
    DRY_RUN=false
fi

# 检查源文件夹和目标文件夹是否存在
if [ ! -d "$RAW_DATA_DIR" ]; then
    echo -e "${RED}错误: 原始数据文件夹 '$RAW_DATA_DIR' 不存在。脚本已终止。${NC}" | tee -a "$LOG_FILE"
    exit 1
fi

if [ ! -d "$PREPROCESSED_DIR" ]; then
    echo -e "${RED}错误: 预处理数据文件夹 '$PREPROCESSED_DIR' 不存在。脚本已终止。${NC}" | tee -a "$LOG_FILE"
    exit 1
fi

# 【新增】检查Metadata CSV文件是否存在
if [ ! -f "$METADATA_CSV" ]; then
    echo -e "${RED}错误: Metadata CSV 文件 '$METADATA_CSV' 不存在。脚本已终止。${NC}" | tee -a "$LOG_FILE"
    exit 1
fi

# 创建归档文件夹 (如果它不存在)
echo "检查归档文件夹 '$DESTINATION_DIR'..." | tee -a "$LOG_FILE"
if [ "$DRY_RUN" = false ]; then
    mkdir -p "$DESTINATION_DIR"
    echo "归档文件夹已确保存在。" | tee -a "$LOG_FILE"
else
    echo "[预演] 将会创建归档文件夹 (如果不存在)。" | tee -a "$LOG_FILE"
fi
echo "--------------------------------------------------" | tee -a "$LOG_FILE"

# 【新增】从CSV表头动态获取 'cal_sky_direction' 所在的列号
header=$(head -n 1 "$METADATA_CSV")
if [ -z "$header" ]; then
    echo -e "${RED}错误: Metadata CSV 文件 '$METADATA_CSV' 为空或无法读取表头。${NC}" | tee -a "$LOG_FILE"
    exit 1
fi
# 通过将逗号转换成换行，利用grep -n获取行号（即列号）
sky_direction_col=$(echo "$header" | tr ',' '\n' | grep -nx "cal_sky_direction" | cut -d: -f1)
if [ -z "$sky_direction_col" ]; then
    echo -e "${RED}错误: 在 '$METADATA_CSV' 中未找到 'cal_sky_direction' 列。${NC}" | tee -a "$LOG_FILE"
    exit 1
fi
echo "信息: 'cal_sky_direction' 是第 ${sky_direction_col} 列。" | tee -a "$LOG_FILE"
echo "--------------------------------------------------" | tee -a "$LOG_FILE"


# 初始化计数器
processed_ids_count=0
moved_count=0
skipped_raw_missing_count=0
skipped_metadata_count=0

# 遍历预处理文件夹中的每一个条目 (场景ID)
for scene_id_path in $(find "$PREPROCESSED_DIR" -maxdepth 1 -mindepth 1 -type d); do
    # 从完整路径中提取场景ID (即文件夹名)
    scene_id=$(basename "$scene_id_path")
    ((processed_ids_count++))

    # 构建对应的原始数据文件夹路径
    source_to_move="$RAW_DATA_DIR/$scene_id"

    # 检查原始数据文件夹是否存在
    if [ -d "$source_to_move" ]; then
        echo -e "${GREEN}[处理中] 找到匹配的原始数据: '$source_to_move'${NC}" | tee -a "$LOG_FILE"

        # 【核心修改】检查metadata条件
        # 使用 grep 精准匹配以 scene_id 开头的行
        line_data=$(grep "^${scene_id}," "$METADATA_CSV")

        if [ -z "$line_data" ]; then
            echo -e "  └── ${YELLOW}[跳过] 在CSV中未找到场景ID '$scene_id' 的记录。${NC}" | tee -a "$LOG_FILE"
            ((skipped_metadata_count++))
        else
            # 使用 awk 和之前获取的列号来提取属性值
            sky_direction_value=$(echo "$line_data" | awk -F',' -v col="$sky_direction_col" '{print $col}')

            # 检查属性值是否为 "Up"
            if [[ "$sky_direction_value" == "Up" ]]; then
                echo -e "  └── ${GREEN}条件满足: '$scene_id' 的 cal_sky_direction 是 'Up'。${NC}" | tee -a "$LOG_FILE"

                if [ "$DRY_RUN" = true ]; then
                    echo "      └── [预演] 将会移动 '$source_to_move' 到 '$DESTINATION_DIR/'" | tee -a "$LOG_FILE"
                else
                    # 执行移动命令
                    mv "$source_to_move" "$DESTINATION_DIR/"
                    echo "      └── ${GREEN}成功移动到 '$DESTINATION_DIR/'${NC}" | tee -a "$LOG_FILE"
                fi
                ((moved_count++))
            else
                echo -e "  └── ${YELLOW}[跳过] 条件不满足: '$scene_id' 的 cal_sky_direction 是 '${sky_direction_value}' (不是 'Up')。${NC}" | tee -a "$LOG_FILE"
                ((skipped_metadata_count++))
            fi
        fi
    else
        # 如果原始数据文件夹不存在 (可能已被移动或删除)，则跳过并记录
        echo -e "${YELLOW}[跳过] 在预处理文件夹中找到 '$scene_id'，但在原始数据文件夹中未找到对应目录。${NC}" | tee -a "$LOG_FILE"
        ((skipped_raw_missing_count++))
    fi
done

echo "--------------------------------------------------" | tee -a "$LOG_FILE"
echo -e "${GREEN}脚本执行完毕！${NC}" | tee -a "$LOG_FILE"
echo "--- 摘要 ---" | tee -a "$LOG_FILE"
echo "共检查了 ${processed_ids_count} 个预处理场景ID。" | tee -a "$LOG_FILE"
echo -e "成功移动 ${moved_count} 个文件夹。" | tee -a "$LOG_FILE"
echo "跳过 ${skipped_raw_missing_count} 个 (因原始数据目录不存在)。" | tee -a "$LOG_FILE"
echo "跳过 ${skipped_metadata_count} 个 (因metadata条件不满足或ID不存在)。" | tee -a "$LOG_FILE"
echo "详细日志请查看文件: ${LOG_FILE}" | tee -a "$LOG_FILE"