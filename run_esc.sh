#!/bin/bash

# ESC 批量评估脚本（进程级并发版本）
# 通过启动多个 Python 进程来加速 API 调用
# 适配：src/runtime/esc_history_principle.py

# ===== 配置参数 =====
MAX_TURNS=9
MAX_SAMPLES=200
START_INDEX=0
END_INDEX=119
BATCH_SIZE=40

ESC_DEV_PATH="data/ESC/test.json"
ESC_PERSONA_PATH="outputs/P4G/personas/personas_eval.jsonl"
CLUSTER_CONFIG="kmeans_k150"

ABLAATION_MODE="none" # none | w/o_depth | w/o_breadth | w/o_both | w/o_expend

OUTPUT_DIR="outputs/ESC/evaluate/${CLUSTER_CONFIG}/history"
OUTPUT_FILE="${OUTPUT_DIR}/${ABLAATION_MODE}/standard.json"

# ===== 计算批次信息 =====
TOTAL_SAMPLES=$((END_INDEX - START_INDEX + 1))
NUM_BATCHES=$(( (TOTAL_SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE ))

echo "=========================================="
echo "ESC 批量评估（进程级并发）"
echo "=========================================="
echo "总样本数: ${TOTAL_SAMPLES}"
echo "批次大小: ${BATCH_SIZE}"
echo "批次数量: ${NUM_BATCHES}"
echo "输出文件: ${OUTPUT_FILE}"
echo "=========================================="

# 确保输出目录存在
mkdir -p "${OUTPUT_DIR}/${ABLAATION_MODE}"

# ===== 信号处理 =====
trap_ctrlc() {
    echo ""
    echo "Ctrl+C detected. Terminating all processes..."
    pkill -P $$
    exit 2
}
trap trap_ctrlc INT

# ===== 启动多个进程 =====
for ((i=0; i<NUM_BATCHES; i++)); do
    BATCH_START=$((START_INDEX + i * BATCH_SIZE))
    BATCH_END=$((BATCH_START + BATCH_SIZE - 1))
    if [ $BATCH_END -gt $END_INDEX ]; then
        BATCH_END=$END_INDEX
    fi

    echo "启动批次 $((i+1))/$NUM_BATCHES: 索引 ${BATCH_START} 到 ${BATCH_END}"

    python src/runtime/history_principle_esc.py \
        --max_turns ${MAX_TURNS} \
        --start_index ${BATCH_START} \
        --end_index ${BATCH_END} \
        --cluster_config "${CLUSTER_CONFIG}" \
        --output_file "${OUTPUT_FILE}" \
        --dev_path "${ESC_DEV_PATH}" \
        --persona_path "${ESC_PERSONA_PATH}" \
        --ablation_mode "${ABLAATION_MODE}" \
        --verbose \
        &
done

echo ""
echo "所有批次已启动，等待完成..."
wait

echo ""
echo "=========================================="
echo "所有批次已完成！"
echo "结果已保存到: ${OUTPUT_FILE}"
echo "=========================================="