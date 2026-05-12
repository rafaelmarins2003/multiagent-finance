#!/usr/bin/env bash
set -u

mkdir -p data/requests data/snapshots data/results data/traces data/logs

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OFFSET="${ENRICHED_OFFSET:-0}"
LIMIT="${ENRICHED_LIMIT:-10}"
NEWS_COUNT="${ENRICHED_NEWS_COUNT:-2}"
PRICE_PERIOD="${ENRICHED_PRICE_PERIOD:-1y}"

REQUEST_DIR="data/requests/enriched_${RUN_ID}"
SNAPSHOT_DIR="data/snapshots/enriched_${RUN_ID}"
mkdir -p "${REQUEST_DIR}" "${SNAPSHOT_DIR}"

echo "=== enriched subset run=${RUN_ID} offset=${OFFSET} limit=${LIMIT} started $(date -Is) ==="

uv run python - <<PY
import json
import re
from pathlib import Path

offset = int("${OFFSET}")
limit = int("${LIMIT}")
request_dir = Path("${REQUEST_DIR}")
data = json.loads(Path("examples/portfolios/portfolios.json").read_text())
selected = data["portfolios"][offset:offset + limit]

for item in selected:
    case_id = item["id"]
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id)
    payload = {
        "case_id": case_id,
        "portfolio": item["portfolio"],
        "profile": item["profile"],
        "metadata": {k: v for k, v in item.items() if k not in {"portfolio", "profile"}},
    }
    (request_dir / f"{safe_id}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\\n",
        encoding="utf-8",
    )
print(f"wrote {len(selected)} requests to {request_dir}")
PY

for request in "${REQUEST_DIR}"/*.json; do
  case_id="$(basename "${request}" .json)"
  snapshot="${SNAPSHOT_DIR}/${case_id}.snapshot.json"

  echo "=== snapshot ${case_id} started $(date -Is) ==="
  if ! uv run timeout 1800s python scripts/generate_snapshot.py \
      --input "${request}" \
      --output "${snapshot}" \
      --price-period "${PRICE_PERIOD}" \
      --news-count "${NEWS_COUNT}" \
      2>&1 | tee "data/logs/enriched_${RUN_ID}_${case_id}_snapshot.log"; then
    echo "=== snapshot ${case_id} failed; skipping baselines ==="
    continue
  fi

  echo "=== baselines enriched ${case_id} started $(date -Is) ==="
  if ! uv run timeout 14400s env \
      PYTHONUNBUFFERED=1 \
      LLM_MAX_CONCURRENCY="${LLM_MAX_CONCURRENCY:-1}" \
      STRUCTURED_OUTPUT_MODE="${STRUCTURED_OUTPUT_MODE:-manual}" \
      STRUCTURED_OUTPUT_MAX_RETRIES="${STRUCTURED_OUTPUT_MAX_RETRIES:-6}" \
      STRUCTURED_OUTPUT_RETRY_BASE_SECONDS="${STRUCTURED_OUTPUT_RETRY_BASE_SECONDS:-20}" \
      STRUCTURED_OUTPUT_RETRY_MAX_SECONDS="${STRUCTURED_OUTPUT_RETRY_MAX_SECONDS:-120}" \
      OLLAMA_TIMEOUT_SECONDS="${OLLAMA_TIMEOUT_SECONDS:-180}" \
      OLLAMA_NUM_PREDICT="${OLLAMA_NUM_PREDICT:-2048}" \
      OLLAMA_REASONING="${OLLAMA_REASONING:-false}" \
      DEBATE_MAX_ROUNDS="${DEBATE_MAX_ROUNDS:-1}" \
      DEBATE_MIN_ROUNDS="${DEBATE_MIN_ROUNDS:-1}" \
      python scripts/run_baselines.py \
        --snapshot "${snapshot}" \
        --case-id "enriched_${case_id}" \
        --baselines b1 b2 b3 b4h b4r \
        --self-consistency-samples 5 \
        --trace-db "data/traces/enriched_${RUN_ID}_${case_id}.sqlite3" \
        --output "data/results/enriched_${RUN_ID}_${case_id}.json" \
      2>&1 | tee "data/logs/enriched_${RUN_ID}_${case_id}_baselines.log"; then
    echo "=== baselines enriched ${case_id} failed ==="
    continue
  fi
done

echo "=== enriched subset summary run=${RUN_ID} ==="
uv run python - <<PY
import json
from pathlib import Path

run_id = "${RUN_ID}"
paths = sorted(Path("data/results").glob(f"enriched_{run_id}_*.json"))
print("result_files", len(paths))
for path in paths:
    data = json.loads(path.read_text())
    print(path.name, data.get("summary", {}))
PY

echo "=== enriched subset finished $(date -Is) run=${RUN_ID} ==="
