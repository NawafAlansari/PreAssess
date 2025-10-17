#!/usr/bin/env bash
# End-to-end helper for refreshing Seattle Municipal Code data.
# Usage: ./data_processing/run_pipeline.sh [path/to/pdf]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data"
PROCESSED_DIR="${DATA_DIR}/processed"

PDF_PATH="${1:-${SCRIPT_DIR}/Seattle, WA Municipal Code.pdf}"
if [[ ! -f "${PDF_PATH}" ]]; then
  echo "✖ PDF not found: ${PDF_PATH}" >&2
  exit 1
fi

mkdir -p "${DATA_DIR}" "${PROCESSED_DIR}"

echo "→ Extracting titles from ${PDF_PATH}"
for TITLE in ${TITLES:-"22 23"}; do
  NEXT_TITLE=$((TITLE + 1))
  OUTPUT_PATH="${DATA_DIR}/title${TITLE}.json"
  python "${SCRIPT_DIR}/extract_title.py" \
    "${PDF_PATH}" \
    --title "${TITLE}" \
    --next-title "${NEXT_TITLE}" \
    --output "${OUTPUT_PATH}"
done

echo "→ Building chunk file"
python "${SCRIPT_DIR}/build_ground_truth.py" \
  "${DATA_DIR}"/title*.json \
  --output "${PROCESSED_DIR}/smc_chunks.jsonl"

echo "→ Building SQLite index"
python "${SCRIPT_DIR}/build_sqlite_index.py" \
  "${PROCESSED_DIR}/smc_chunks.jsonl" \
  --database "${PROCESSED_DIR}/smc_ground_truth.db"

echo "→ Generating embeddings"
python "${SCRIPT_DIR}/build_embeddings.py" \
  --chunks "${PROCESSED_DIR}/smc_chunks.jsonl" \
  --output "${PROCESSED_DIR}/smc_embeddings.npz"

echo "✓ Data pipeline complete."
