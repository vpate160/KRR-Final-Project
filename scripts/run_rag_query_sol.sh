#!/bin/bash
#SBATCH --job-name=rag_query_llama31
#SBATCH --output=logs/rag_query_%j.out
#SBATCH --error=logs/rag_query_%j.err
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00

set -euo pipefail

# Resolve project root as parent directory of this script's folder.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/.venv}"

mkdir -p "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"

echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Using venv: ${VENV_PATH}"

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "[ERROR] Python environment not found at ${VENV_PATH}."
  echo "[ERROR] Set VENV_PATH or create .venv before submitting this job."
  exit 1
fi

# Activate environment
source "${VENV_PATH}/bin/activate"

# Optional: if your cluster uses modules, uncomment and adapt these lines.
# module purge
# module load cuda/12.1

# Hugging Face auth check for gated Llama model.
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[WARN] HF_TOKEN is not set in the environment."
  echo "[WARN] The job will only work if `huggingface-cli login` was previously done on this account/node."
fi

python "scripts/build_rag_pipeline.py" \
  --vectorstore-path "data/vectorstore/clean" \
  --collection-name "kb" \
  --model-path "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --question "What are the main causes and effects of the Great Depression?"
