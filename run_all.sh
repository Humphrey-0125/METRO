#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_all.sh --dataset {p4g|cb|esc} [options]

Options:
  --dataset         Dataset name: p4g | cb | esc (required)
  --max-turns       Max dialogue turns (default: 9)
  --max-samples     Max samples for python runtime (default: dataset specific)
  --start-index     Start index (default: dataset specific)
  --end-index       End index (default: dataset specific)
  --batch-size      Batch size for process-level parallel batching
  --cluster-config  Cluster config (default: dataset specific)
  --ablation-mode   none | w/o_depth | w/o_breadth | w/o_both | w/o_expend (default: none)
  --output-file     Output json path (default: dataset specific)
  --dev-path        Override dev/test path (cb/esc only)
  --persona-path    Override persona path (cb/esc only)
  --max-parallel    Max concurrent python processes, 0 means unlimited (default: 0)
  --verbose         Pass --verbose to python runtime
  -h, --help        Show this help
EOF
}

DATASET=""
MAX_TURNS=9
MAX_SAMPLES=""
START_INDEX=""
END_INDEX=""
BATCH_SIZE=""
CLUSTER_CONFIG=""
ABLATION_MODE="none"
OUTPUT_FILE=""
DEV_PATH=""
PERSONA_PATH=""
MAX_PARALLEL=0
VERBOSE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="${2:-}"; shift 2 ;;
    --max-turns) MAX_TURNS="${2:-}"; shift 2 ;;
    --max-samples) MAX_SAMPLES="${2:-}"; shift 2 ;;
    --start-index) START_INDEX="${2:-}"; shift 2 ;;
    --end-index) END_INDEX="${2:-}"; shift 2 ;;
    --batch-size) BATCH_SIZE="${2:-}"; shift 2 ;;
    --cluster-config) CLUSTER_CONFIG="${2:-}"; shift 2 ;;
    --ablation-mode) ABLATION_MODE="${2:-}"; shift 2 ;;
    --output-file) OUTPUT_FILE="${2:-}"; shift 2 ;;
    --dev-path) DEV_PATH="${2:-}"; shift 2 ;;
    --persona-path) PERSONA_PATH="${2:-}"; shift 2 ;;
    --max-parallel) MAX_PARALLEL="${2:-}"; shift 2 ;;
    --verbose) VERBOSE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${DATASET}" ]]; then
  echo "Error: --dataset is required."
  usage
  exit 1
fi

RUNTIME=""
case "${DATASET}" in
  p4g)
    RUNTIME="src/runtime/history_principle.py"
    [[ -z "${MAX_SAMPLES}" ]] && MAX_SAMPLES=200
    [[ -z "${START_INDEX}" ]] && START_INDEX=0
    [[ -z "${END_INDEX}" ]] && END_INDEX=1
    [[ -z "${BATCH_SIZE}" ]] && BATCH_SIZE=20
    [[ -z "${CLUSTER_CONFIG}" ]] && CLUSTER_CONFIG="kmeans_k150"
    [[ -z "${OUTPUT_FILE}" ]] && OUTPUT_FILE="outputs/P4G/evaluate/history/${CLUSTER_CONFIG}/none/test.json"
    ;;
  cb)
    RUNTIME="src/runtime/history_principle_cb.py"
    [[ -z "${MAX_SAMPLES}" ]] && MAX_SAMPLES=200
    [[ -z "${START_INDEX}" ]] && START_INDEX=0
    [[ -z "${END_INDEX}" ]] && END_INDEX=199
    [[ -z "${BATCH_SIZE}" ]] && BATCH_SIZE=15
    [[ -z "${CLUSTER_CONFIG}" ]] && CLUSTER_CONFIG="kmeans_k80"
    [[ -z "${DEV_PATH}" ]] && DEV_PATH="data/CB/dev.json"
    [[ -z "${PERSONA_PATH}" ]] && PERSONA_PATH="outputs/P4G/personas/personas_eval.jsonl"
    [[ -z "${OUTPUT_FILE}" ]] && OUTPUT_FILE="outputs/CB/evaluate/${CLUSTER_CONFIG}/history/${ABLATION_MODE}/depth3.json"
    ;;
  esc)
    RUNTIME="src/runtime/history_principle_esc.py"
    [[ -z "${MAX_SAMPLES}" ]] && MAX_SAMPLES=200
    [[ -z "${START_INDEX}" ]] && START_INDEX=0
    [[ -z "${END_INDEX}" ]] && END_INDEX=119
    [[ -z "${BATCH_SIZE}" ]] && BATCH_SIZE=40
    [[ -z "${CLUSTER_CONFIG}" ]] && CLUSTER_CONFIG="kmeans_k150"
    [[ -z "${DEV_PATH}" ]] && DEV_PATH="data/ESC/test.json"
    [[ -z "${PERSONA_PATH}" ]] && PERSONA_PATH="outputs/P4G/personas/personas_eval.jsonl"
    [[ -z "${OUTPUT_FILE}" ]] && OUTPUT_FILE="outputs/ESC/evaluate/${CLUSTER_CONFIG}/history/${ABLATION_MODE}/standard.json"
    ;;
  *)
    echo "Error: unsupported dataset '${DATASET}'. Use p4g|cb|esc."
    exit 1
    ;;
esac

if [[ "${END_INDEX}" -lt "${START_INDEX}" ]]; then
  echo "Error: end_index (${END_INDEX}) must be >= start_index (${START_INDEX})."
  exit 1
fi

TOTAL_SAMPLES=$((END_INDEX - START_INDEX + 1))
NUM_BATCHES=$(((TOTAL_SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE))
OUTPUT_DIR="$(dirname "${OUTPUT_FILE}")"
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Unified batch runner"
echo "dataset:      ${DATASET}"
echo "runtime:      ${RUNTIME}"
echo "range:        ${START_INDEX}..${END_INDEX} (total ${TOTAL_SAMPLES})"
echo "batch_size:   ${BATCH_SIZE}"
echo "num_batches:  ${NUM_BATCHES}"
echo "cluster:      ${CLUSTER_CONFIG}"
echo "ablation:     ${ABLATION_MODE}"
echo "output:       ${OUTPUT_FILE}"
echo "max_parallel: ${MAX_PARALLEL}"
echo "=========================================="

trap_ctrlc() {
  echo ""
  echo "Ctrl+C detected. Terminating all processes..."
  pkill -P $$ || true
  exit 2
}
trap trap_ctrlc INT

wait_for_slot() {
  if [[ "${MAX_PARALLEL}" -le 0 ]]; then
    return 0
  fi
  while true; do
    running=$(jobs -pr | wc -l | tr -d ' ')
    if [[ "${running}" -lt "${MAX_PARALLEL}" ]]; then
      break
    fi
    wait -n || true
  done
}

for ((i=0; i<NUM_BATCHES; i++)); do
  BATCH_START=$((START_INDEX + i * BATCH_SIZE))
  BATCH_END=$((BATCH_START + BATCH_SIZE - 1))
  if [[ "${BATCH_END}" -gt "${END_INDEX}" ]]; then
    BATCH_END="${END_INDEX}"
  fi

  echo "Launching batch $((i+1))/${NUM_BATCHES}: ${BATCH_START}..${BATCH_END}"
  wait_for_slot

  cmd=(python "${RUNTIME}"
    --max_turns "${MAX_TURNS}"
    --max_samples "${MAX_SAMPLES}"
    --start_index "${BATCH_START}"
    --end_index "${BATCH_END}"
    --cluster_config "${CLUSTER_CONFIG}"
    --output_file "${OUTPUT_FILE}"
    --ablation_mode "${ABLATION_MODE}"
  )

  if [[ "${VERBOSE}" -eq 1 ]]; then
    cmd+=(--verbose)
  fi

  if [[ "${DATASET}" == "cb" || "${DATASET}" == "esc" ]]; then
    cmd+=(--dev_path "${DEV_PATH}" --persona_path "${PERSONA_PATH}")
  fi

  "${cmd[@]}" &
done

echo ""
echo "All batches started. Waiting for completion..."
wait

echo ""
echo "=========================================="
echo "All batches completed."
echo "Results saved to: ${OUTPUT_FILE}"
echo "=========================================="
