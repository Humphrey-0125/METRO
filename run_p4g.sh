#!/bin/bash
set -euo pipefail

# History Principle 批量评估脚本（进程级并发版本）
# 支持消融参数：--ablation_mode {none|w/o_depth|w/o_breadth|w/o_both}

# ===== 配置参数 =====
MAX_TURNS=9
MAX_SAMPLES=200                       # 评估样本上限（由 Python 侧决定 n=min(max_samples, dataset_len, ...)）
CLUSTER_CONFIG="kmeans_k150"

# 批次处理配置（按 index 切分）
START_INDEX=0
END_INDEX=1
BATCH_SIZE=20

# 消融配置：none | w/o_depth | w/o_breadth | w/o_both | w/o_expend
ABLATION_MODE="none"

# 进程并发上限（可选；0 表示不限制）
MAX_PARALLEL=0

# ===== 输出文件配置 =====
TIMESTAMP=$(TZ='Asia/Seoul' date "+%Y%m%d_%H%M%S")
# OUTPUT_DIR="outputs/P4G/evaluate/history/${CLUSTER_CONFIG}/new_metrics"
# SAFE_MODE="${ABLATION_MODE//\//_}"   # 把 w/o_depth -> w_o_depth 这种形式更安全（虽然这里不含 / 也没事）
# OUTPUT_FILE="/${OUTPUT_DIR}/${SAFE_MODE}/results_Ours_1_personas_${TIMESTAMP}.json"
OUTPUT_DIR="outputs/P4G/evaluate/history"
OUTPUT_FILE="${OUTPUT_DIR}/${CLUSTER_CONFIG}/none/test.json"

# ===== 计算批次信息 =====
TOTAL_SAMPLES=$((END_INDEX - START_INDEX + 1))
NUM_BATCHES=$(( (TOTAL_SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE ))

echo "=========================================="
echo "History Principle 批量评估（进程级并发）"
echo "=========================================="
echo "总样本数:     ${TOTAL_SAMPLES} (index ${START_INDEX}..${END_INDEX})"
echo "max_samples:  ${MAX_SAMPLES}"
echo "批次大小:     ${BATCH_SIZE}"
echo "批次数量:     ${NUM_BATCHES}"
echo "cluster:      ${CLUSTER_CONFIG}"
echo "ablation:     ${ABLATION_MODE}"
echo "输出文件:     ${OUTPUT_FILE}"
echo "=========================================="

mkdir -p "${OUTPUT_DIR}"

# ===== 信号处理 =====
trap_ctrlc() {
  echo ""
  echo "Ctrl+C detected. Terminating all processes..."
  pkill -P $$ || true
  exit 2
}
trap trap_ctrlc INT

# ===== 并发控制（可选） =====
# 若 MAX_PARALLEL > 0，则当后台作业数达到上限时等待一个完成
wait_for_slot() {
  if [ "${MAX_PARALLEL}" -le 0 ]; then
    return 0
  fi
  while true; do
    running=$(jobs -pr | wc -l | tr -d ' ')
    if [ "${running}" -lt "${MAX_PARALLEL}" ]; then
      break
    fi
    # 等待任意一个任务结束（bash 5+ 支持 wait -n）
    wait -n || true
  done
}

# ===== 启动多个进程 =====
for ((i=0; i<NUM_BATCHES; i++)); do
  BATCH_START=$((START_INDEX + i * BATCH_SIZE))
  BATCH_END=$((BATCH_START + BATCH_SIZE - 1))
  if [ "${BATCH_END}" -gt "${END_INDEX}" ]; then
    BATCH_END="${END_INDEX}"
  fi

  echo "启动批次 $((i+1))/${NUM_BATCHES}: 索引 ${BATCH_START} 到 ${BATCH_END}"

  wait_for_slot

  python src/runtime/history_principle.py \
    --max_turns "${MAX_TURNS}" \
    --max_samples "${MAX_SAMPLES}" \
    --cluster_config "${CLUSTER_CONFIG}" \
    --start_index "${BATCH_START}" \
    --end_index "${BATCH_END}" \
    --output_file "${OUTPUT_FILE}" \
    --ablation_mode "${ABLATION_MODE}" \
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
