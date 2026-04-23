#!/usr/bin/env bash
# Sweep sglang.bench_serving over --max-concurrency values.
# Results: bench_serving writes each run as one JSON line (JSONL) to --output-file.
# This script also merges those lines into a single JSON array file.
#
# Requires a running SGLang server; set BASE_URL if not default.
#
# Usage:
#   ./benchmark/bench_serving_glm5_conc_sweep.sh
#   OUT_DIR=./results ./benchmark/bench_serving_glm5_conc_sweep.sh

set -euo pipefail

OUT_DIR="${OUT_DIR:-.}"
mkdir -p "${OUT_DIR}"

PORT="${PORT:-30000}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
JSONL_OUT="${OUT_DIR}/bench_glm5_fp8_conc_sweep_${STAMP}.jsonl"
JSON_OUT="${OUT_DIR}/bench_glm5_fp8_conc_sweep_${STAMP}.json"
# Optional: export BASE_URL=http://127.0.0.1:30000 if the server is not on defaults.

# Truncate so repeated runs with the same STAMP do not append to old data
: > "${JSONL_OUT}"

for CONC in 4 8 16 32; do
  NUM_PROMPTS=$((CONC * 8))
  echo "== max_concurrency=${CONC} num_prompts=${NUM_PROMPTS} =="
  CMD=(python3 -m sglang.bench_serving --backend sglang)
  if [[ -n "${BASE_URL:-}" ]]; then
    CMD+=(--base-url "${BASE_URL}")
  fi
  CMD+=(
    --model zai-org/GLM-5-FP8
    --dataset-name random
    --random-input-len 1000
    --random-output-len 1000
    --num-prompts "${NUM_PROMPTS}"
    --max-concurrency "${CONC}"
    --request-rate inf
    --output-file "${JSONL_OUT}"
    --port "${PORT}"
  )
  "${CMD[@]}"
done

# Merge JSONL (one object per line) into a single JSON array file
python3 - <<'PY' "${JSONL_OUT}" "${JSON_OUT}"
import json, sys
src, dst = sys.argv[1], sys.argv[2]
rows = []
with open(src, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
with open(dst, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)
    f.write("\n")
print(f"Wrote merged JSON: {dst}")
PY

echo "Per-run lines (JSONL): ${JSONL_OUT}"
echo "Merged array (JSON):  ${JSON_OUT}"
