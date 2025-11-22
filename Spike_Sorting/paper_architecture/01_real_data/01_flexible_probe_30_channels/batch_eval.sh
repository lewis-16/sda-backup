#!/bin/bash
# 批量运行evaluation脚本
# 对sorting_new目录下除了021322外的所有月份进行evaluation

# 设置基础路径
BASE_DIR="/media/ubuntu/sda/Spike_Sorting/paper_architecture/01_real_data/01_flexible_probe_30_channels"
SORTING_NEW_DIR="${BASE_DIR}/kilosort_spike_sorting/sorting_new"
EVAL_SCRIPT="${BASE_DIR}/eval_spike_pipeline.py"
SKIP_DATE="021322"  # 跳过的日期（训练集）

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}批量运行Spike Sorting Evaluation${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}注意: 如果neuron_inf.pkl是用旧版本生成的（31维waveform），${NC}"
echo -e "${YELLOW}      请先运行generate_neuron_inf_phy.py重新生成（30维waveform）${NC}"
echo ""

# 检查eval脚本是否存在
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo -e "${RED}错误: 找不到eval脚本: $EVAL_SCRIPT${NC}"
    exit 1
fi

# 检查sorting_new目录是否存在
if [ ! -d "$SORTING_NEW_DIR" ]; then
    echo -e "${RED}错误: 找不到sorting_new目录: $SORTING_NEW_DIR${NC}"
    exit 1
fi

# 获取所有日期目录（排除021322）
dates=()
for dir in "$SORTING_NEW_DIR"/*; do
    if [ -d "$dir" ]; then
        date=$(basename "$dir")
        if [ "$date" != "$SKIP_DATE" ]; then
            dates+=("$date")
        fi
    fi
done

# 按日期排序（使用数组排序）
if [ ${#dates[@]} -gt 0 ]; then
    IFS=$'\n' sorted_dates=($(sort <<<"${dates[*]}"))
    unset IFS
    dates=("${sorted_dates[@]}")
fi

echo -e "${YELLOW}找到 ${#dates[@]} 个日期需要处理:${NC}"
for date in "${dates[@]}"; do
    echo "  - $date"
done
echo ""

# 询问确认
read -p "是否继续? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# 记录开始时间
start_time=$(date +%s)
total_dates=${#dates[@]}
current=0

# 遍历每个日期
for date in "${dates[@]}"; do
    current=$((current + 1))
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}[$current/$total_dates] 处理日期: $date${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    # 检查spike_inf.tsv是否存在
    spike_inf_path="${SORTING_NEW_DIR}/${date}/spike_inf.tsv"
    if [ ! -f "$spike_inf_path" ]; then
        echo -e "${YELLOW}警告: 找不到spike_inf.tsv: $spike_inf_path${NC}"
        echo -e "${YELLOW}跳过该日期${NC}"
        continue
    fi
    
    # 检查录音文件是否存在
    recording_path="/media/ubuntu/sda/data/mouse6/ns4/natural_image/mouse6_${date}_natural_image_001.ns4"
    if [ ! -f "$recording_path" ]; then
        echo -e "${YELLOW}警告: 找不到录音文件: $recording_path${NC}"
        echo -e "${YELLOW}跳过该日期${NC}"
        continue
    fi
    
    # 运行evaluation
    echo "开始运行evaluation..."
    python "$EVAL_SCRIPT" --date "$date" --base-dir "$BASE_DIR"
    
    # 检查返回状态
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ 日期 $date 处理完成${NC}"
    else
        echo -e "${RED}✗ 日期 $date 处理失败${NC}"
    fi
    
    # 显示进度
    elapsed=$(($(date +%s) - start_time))
    avg_time=$((elapsed / current))
    remaining=$((avg_time * (total_dates - current)))
    echo "已用时间: ${elapsed}秒 | 预计剩余: ${remaining}秒"
done

# 计算总时间
total_time=$(($(date +%s) - start_time))
hours=$((total_time / 3600))
minutes=$(((total_time % 3600) / 60))
seconds=$((total_time % 60))

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}所有处理完成!${NC}"
echo -e "${GREEN}总用时: ${hours}小时 ${minutes}分钟 ${seconds}秒${NC}"
echo -e "${GREEN}========================================${NC}"

