#!/bin/bash

# CB 批量评估脚本（进程级并发版本）
# 通过启动多个 Python 进程来加速 API 调用

# ===== 配置参数 =====
MAX_TURNS=9
MAX_SAMPLES=200   # 总样本数（CB: 200）
START_INDEX=0
END_INDEX=199
BATCH_SIZE=15     # 每个进程处理的样本数

# （可选）数据路径：如果你的 simulate_cb.py 支持这些 CLI 参数就保留；不支持就删掉
CB_DEV_PATH="data/CB/dev.json"
CB_PERSONA_PATH="outputs/P4G/personas/personas_eval.jsonl"
CLUSTER_CONFIG="kmeans_k80"
# 输出文件配置（所有进程共享同一个输出文件，通过增量保存机制保证线程安全）
TIMESTAMP=$(TZ='Asia/Seoul' date "+%Y%m%d_%H%M%S")

# 你可以按你的目录习惯改：raw_model / history_principle / ours 等
ABLAATION_MODE="none" # none | w/o_depth | w/o_breadth | w/o_both | w/o_expend
OUTPUT_DIR="outputs/CB/evaluate/${CLUSTER_CONFIG}/history"
OUTPUT_FILE="${OUTPUT_DIR}/${ABLAATION_MODE}/depth3.json"
# OUTPUT_DIR="outputs/CB/evaluate/history"
# OUTPUT_FILE="${OUTPUT_DIR}/p2c/plt_results_${TIMESTAMP}.json"
# ===== 计算批次信息 =====
TOTAL_SAMPLES=$((END_INDEX - START_INDEX + 1))
NUM_BATCHES=$(( (TOTAL_SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE ))

echo "=========================================="
echo "CB 批量评估（进程级并发）"
echo "=========================================="
echo "总样本数: ${TOTAL_SAMPLES}"
echo "批次大小: ${BATCH_SIZE}"
echo "批次数量: ${NUM_BATCHES}"
echo "输出文件: ${OUTPUT_FILE}"
echo "=========================================="

# 确保输出目录存在
mkdir -p "${OUTPUT_DIR}"

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

    # 这里的脚本名按你的实际文件改：
    # - 如果你跑 raw baseline：src/runtime/simulate_cb.py（或 raw_model_cb.py）
    # - 如果你跑 history+principle：src/runtime/history_principle_cb.py（你即将改的那个）
    python src/runtime/history_principle_cb.py \
        --max_turns ${MAX_TURNS} \
        --start_index ${BATCH_START} \
        --cluster_config "${CLUSTER_CONFIG}" \
        --end_index ${BATCH_END} \
        --output_file "${OUTPUT_FILE}" \
        --dev_path "${CB_DEV_PATH}" \
        --persona_path "${CB_PERSONA_PATH}" \
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