#!/bin/bash
set -euo pipefail

# ============================================================
# Unified batch evaluation script for P4G / CB / ESC
# Usage: bash run.sh
# Change DATASET below to select which dataset to run.
# ============================================================

# ===== Dataset selection =====
# Choose: p4g | cb | esc
DATASET="p4g"

# ===== Common parameters =====
MAX_TURNS=9
MAX_PARALLEL=0    # max concurrent processes; 0 = unlimited

# ===== Dataset-specific parameters =====
case "${DATASET}" in
  p4g)
    START_INDEX=0
    END_INDEX=199
    BATCH_SIZE=20
    MAX_SAMPLES=200
    CLUSTER_CONFIG="kmeans_k150"
    PYTHON_SCRIPT="src/runtime/metro_p4g.py"
    OUTPUT_DIR="outputs/P4G/evaluate/history/${CLUSTER_CONFIG}"
    EXTRA_ARGS=()
    ;;
  cb)
    START_INDEX=0
    END_INDEX=199
    BATCH_SIZE=15
    MAX_SAMPLES=200
    CLUSTER_CONFIG="kmeans_k80"
    PYTHON_SCRIPT="src/runtime/metro_cb.py"
    OUTPUT_DIR="outputs/CB/evaluate/history/${CLUSTER_CONFIG}"
    EXTRA_ARGS=(
      --dev_path "data/CB/dev.json"
      --persona_path "outputs/P4G/personas/personas_eval.jsonl"
    )
    ;;
  esc)
    START_INDEX=0
    END_INDEX=119
    BATCH_SIZE=40
    MAX_SAMPLES=200
    CLUSTER_CONFIG="kmeans_k150"
    PYTHON_SCRIPT="src/runtime/metro_esc.py"
    OUTPUT_DIR="outputs/ESC/evaluate/history/${CLUSTER_CONFIG}"
    EXTRA_ARGS=(
      --dev_path "data/ESC/test.json"
      --persona_path "outputs/P4G/personas/personas_eval.jsonl"
    )
    ;;
  *)
    echo "Unknown dataset '${DATASET}'. Choose: p4g | cb | esc"
    exit 1
    ;;
esac

# ===== Output file =====
TIMESTAMP=$(TZ='Asia/Seoul' date "+%Y%m%d_%H%M%S")
OUTPUT_FILE="${OUTPUT_DIR}/results_${TIMESTAMP}.json"

# ===== Batch calculation =====
TOTAL_SAMPLES=$((END_INDEX - START_INDEX + 1))
NUM_BATCHES=$(( (TOTAL_SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE ))

echo "=========================================="
echo "Dataset: ${DATASET^^}  |  Cluster: ${CLUSTER_CONFIG}"
echo "Samples: ${TOTAL_SAMPLES} (index ${START_INDEX}..${END_INDEX})"
echo "Batches: ${NUM_BATCHES} x ${BATCH_SIZE}"
echo "Output:  ${OUTPUT_FILE}"
echo "=========================================="

mkdir -p "${OUTPUT_DIR}"

# ===== Graceful shutdown on Ctrl+C =====
trap_ctrlc() {
    echo ""
    echo "Ctrl+C — terminating all child processes..."
    pkill -P $$ || true
    exit 2
}
trap trap_ctrlc INT

# ===== Concurrency limiter =====
wait_for_slot() {
    [ "${MAX_PARALLEL}" -le 0 ] && return 0
    while true; do
        running=$(jobs -pr | wc -l | tr -d ' ')
        [ "${running}" -lt "${MAX_PARALLEL}" ] && break
        wait -n || true
    done
}

# ===== Launch one process per batch =====
for ((i=0; i<NUM_BATCHES; i++)); do
    BATCH_START=$((START_INDEX + i * BATCH_SIZE))
    BATCH_END=$((BATCH_START + BATCH_SIZE - 1))
    [ "${BATCH_END}" -gt "${END_INDEX}" ] && BATCH_END="${END_INDEX}"

    echo "Launching batch $((i+1))/${NUM_BATCHES}: index ${BATCH_START}..${BATCH_END}"
    wait_for_slot

    python "${PYTHON_SCRIPT}" \
        --max_turns   "${MAX_TURNS}" \
        --max_samples "${MAX_SAMPLES}" \
        --cluster_config "${CLUSTER_CONFIG}" \
        --start_index "${BATCH_START}" \
        --end_index   "${BATCH_END}" \
        --output_file "${OUTPUT_FILE}" \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
        --verbose \
        &
done

echo ""
echo "All batches launched. Waiting for completion..."
wait

echo ""
echo "=========================================="
echo "Done! Results saved to: ${OUTPUT_FILE}"
echo "=========================================="
