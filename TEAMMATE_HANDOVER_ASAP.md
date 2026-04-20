# Module 3 — Teammate Handover (ASAP Start)

This is a practical handover so you can start immediately with the current Module 3 baseline.

## What is ready right now

- Task 3.1: `scripts/build_kb.py`
- Task 3.2: `scripts/build_vectorstore.py`
- Task 3.3: `scripts/build_rag_pipeline.py` (supports `ollama` backend with `llama3.1:8b`)
- Task 3.4: `scripts/evaluate_rag.py` (EM + token-F1 on NQ)
- Handoff summary: `README_Module3_Handoff.md`

## Baseline status (important)

- Pipeline is runtime-stable locally.
- Retrieval flow is stable and should remain unchanged for baseline comparisons.
- Evaluation quality is still **provisional baseline** (not final tuned quality).
- Latest 20-example check after short-answer prompt tightening:
  - `EM = 0.0500`
  - `token_F1 = 0.1091`

## Fast setup (Apple Silicon / macOS)

```zsh
cd KRR-Final-Project
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install datasets tqdm chromadb sentence-transformers langchain-core langchain-huggingface transformers accelerate
```

### Ollama setup

```zsh
brew install --cask ollama
open -a Ollama
ollama pull llama3.1:8b
```

## Quick sanity run (Task 3.3)

```zsh
python scripts/build_rag_pipeline.py \
  --vectorstore-path data/vectorstore/clean \
  --collection-name kb \
  --generation-backend ollama \
  --ollama-model llama3.1:8b \
  --question "What are the main causes and effects of the Great Depression?"
```

## Evaluation run (Task 3.4)

### Smoke test first (20)

```zsh
python scripts/evaluate_rag.py \
  --vectorstore-path data/vectorstore/clean \
  --collection-name kb \
  --generation-backend ollama \
  --ollama-model llama3.1:8b \
  --n-examples 20 \
  --metrics-path results/metrics.csv \
  --predictions-path results/predictions_clean_20.jsonl
```

### Then scale

```zsh
python scripts/evaluate_rag.py \
  --vectorstore-path data/vectorstore/clean \
  --collection-name kb \
  --generation-backend ollama \
  --ollama-model llama3.1:8b \
  --n-examples 1000 \
  --metrics-path results/metrics.csv \
  --predictions-path results/predictions_clean.jsonl
```

## Artifact sharing expectations

`data/vectorstore/clean/` is intentionally not tracked in Git (large binary index).

Fastest way to reproduce exact behavior:
1. Get `data/clean_kb/kb.jsonl` from repo.
2. Receive `data/vectorstore/clean/` as a zipped artifact from teammate.
3. Place it at `data/vectorstore/clean/`.

Alternative (slower but self-contained): rebuild vectorstore from KB:

```zsh
python scripts/build_vectorstore.py \
  --kb-path data/clean_kb/kb.jsonl \
  --output-dir data/vectorstore/clean \
  --collection-name kb
```

## Guardrails for team consistency

- Do **not** change KB or retrieval logic while measuring clean baseline.
- Keep prompt and evaluation settings frozen while collecting comparable metrics.
- Record every run in `results/metrics.csv` and note model/backend used.

## Suggested immediate split of work

- Person A: run 100 and 1000 evals; track runtime and metrics.
- Person B: inspect prediction errors (answer span mismatch vs. gold).
- Person C: prep Task 3.5+ experiment scaffolding without touching clean baseline scripts.

## Short message you can paste to teammates

"Module 3 baseline is ready in repo. Start from `TEAMMATE_HANDOVER_ASAP.md` and run the 20-example smoke eval first. Keep retrieval unchanged for baseline comparability. If you need exact runtime parity, ask me for the zipped `data/vectorstore/clean/` artifact."
