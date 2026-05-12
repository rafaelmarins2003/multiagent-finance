#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/results data/traces data/logs

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"

run_batch() {
  local label="$1"
  local offset="$2"
  local limit="$3"

  echo "=== ${label} offset=${offset} limit=${limit} started $(date -Is) ==="

  uv run timeout 14400s env \
    PYTHONUNBUFFERED=1 \
    LLM_MAX_CONCURRENCY="${LLM_MAX_CONCURRENCY:-3}" \
    STRUCTURED_OUTPUT_MODE="${STRUCTURED_OUTPUT_MODE:-manual}" \
    STRUCTURED_OUTPUT_MAX_RETRIES="${STRUCTURED_OUTPUT_MAX_RETRIES:-2}" \
    OLLAMA_TIMEOUT_SECONDS="${OLLAMA_TIMEOUT_SECONDS:-180}" \
    OLLAMA_NUM_PREDICT="${OLLAMA_NUM_PREDICT:-2048}" \
    OLLAMA_REASONING="${OLLAMA_REASONING:-false}" \
    DEBATE_MAX_ROUNDS="${DEBATE_MAX_ROUNDS:-1}" \
    DEBATE_MIN_ROUNDS="${DEBATE_MIN_ROUNDS:-1}" \
    python scripts/run_baselines.py \
      --workload examples/portfolios/portfolios.json \
      --baselines b1 b2 b3 b4h b4r \
      --offset-cases "${offset}" \
      --limit-cases "${limit}" \
      --self-consistency-samples 5 \
      --trace-db "data/traces/overnight_${RUN_ID}_${label}.sqlite3" \
      --output "data/results/overnight_${RUN_ID}_${label}.json" \
    2>&1 | tee "data/logs/overnight_${RUN_ID}_${label}.log"
}

echo "=== overnight remaining run=${RUN_ID} started $(date -Is) ==="

# Batch 01 from the previous run is complete. Re-run batch 02 because it had
# parsing errors, and batch 03 because it stopped before writing JSON.
run_batch "batch_02_rerun" 10 10
run_batch "batch_03_rerun" 20 10
run_batch "batch_04" 30 10
run_batch "batch_05" 40 10

echo "=== summary for run=${RUN_ID} ==="
uv run python - <<PY
import json
from pathlib import Path

run_id = "${RUN_ID}"
for path in sorted(Path("data/results").glob(f"overnight_{run_id}_*.json")):
    data = json.loads(path.read_text())
    print(path.name)
    print(json.dumps(data.get("summary", {}), ensure_ascii=False, indent=2))
PY

echo "=== overnight remaining finished $(date -Is) run=${RUN_ID} ==="
