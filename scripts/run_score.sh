#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

# =========================
# Подключить env (если есть)
# =========================
if [ -f "$(dirname "$0")/env.sh" ]; then
  # shellcheck disable=SC1091
  source "$(dirname "$0")/env.sh"
fi

# =========================
# Python
# =========================
PYTHON="${PYTHON:-python}"

echo "Using Python: $PYTHON"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "ERROR: Python not found"
  exit 1
fi

# =========================
# Аргументы
# =========================
INPUT="${1:-data/raw/dataset_2025-03-01_2026-03-29_external.csv}"
OUTPUT="${2:-report/scores.csv}"

if [ -z "$INPUT" ]; then
  echo "ERROR: INPUT is required"
  echo "Usage: run_score.sh input.csv output.csv"
  exit 1
fi

if [ -z "$OUTPUT" ]; then
  echo "ERROR: OUTPUT is required"
  echo "Usage: run_score.sh input.csv output.csv"
  exit 1
fi

echo "Input: $INPUT"
echo "Output: $OUTPUT"

# =========================
# Запуск
# =========================
"$PYTHON" src/score.py --input "$INPUT" --output "$OUTPUT"

echo "Done."