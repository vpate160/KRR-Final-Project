#!/bin/bash
#SBATCH --job-name=module2_detection
#SBATCH --output=logs/module2_%j.out
#SBATCH --error=logs/module2_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE2_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${MODULE2_DIR}/.." && pwd)"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv}"

mkdir -p "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"

echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Module 2 dir:  ${MODULE2_DIR}"
echo "[INFO] Using venv:   ${VENV_PATH}"

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "[ERROR] Python environment not found at ${VENV_PATH}."
  echo "[ERROR] Set VENV_PATH or create .venv before submitting this job."
  exit 1
fi

source "${VENV_PATH}/bin/activate"

if [[ ! -f "${PROJECT_ROOT}/data/embeddings/clean_embeddings.npy" ]]; then
  echo "[ERROR] Clean embeddings missing at data/embeddings/clean_embeddings.npy."
  echo "[ERROR] Pull Module 3's clean artifacts before running Module 2."
  exit 2
fi

POISONED_DIR="${PROJECT_ROOT}/data/poisoned_kb"
if ! ls "${POISONED_DIR}"/*.jsonl >/dev/null 2>&1; then
  echo "[ERROR] No poisoned KB variants in ${POISONED_DIR}."
  echo "[ERROR] git fetch hardik && git checkout hardik/main -- data/poisoned_kb/ logs/"
  exit 3
fi

cd "${MODULE2_DIR}"

echo
echo "=== Task 2.1: encode poisoned KB variants ==="
python -m scripts.run_2_1_extract

echo
echo "=== Task 2.2: unsupervised anomaly detectors (IF + LOF) ==="
python -m scripts.run_2_2_anomaly

echo
echo "=== Task 2.3: supervised MLP classifier ==="
python -m scripts.run_2_3_neural

echo
echo "=== Task 2.4a: perplexity baseline (GPT-2) ==="
python -m scripts.run_2_4_perplexity

echo
echo "[INFO] Tasks 2.1 through 2.4-perplexity complete."
echo "[INFO] Skipped on this job (need Ollama):"
echo "         python -m scripts.run_2_4_llm_judge"
echo "         python -m scripts.run_2_5_mitigate --variant <name> --run-undefended"
echo "[INFO] Detection metrics: ${MODULE2_DIR}/results/detection_metrics.csv"
echo "[INFO] Per-variant scores: ${MODULE2_DIR}/results/scores/"
