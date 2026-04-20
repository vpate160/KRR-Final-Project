# Module 3 Handoff (Tasks 3.1–3.4)

## Scope
This handoff covers Module 3 implementation and local validation work for:
- Task 3.1: Clean Wikipedia KB curation
- Task 3.2: Clean Chroma vector store build
- Task 3.3: Clean RAG inference pipeline (local Ollama path validated)
- Task 3.4: Clean baseline evaluation on Natural Questions (EM / token-F1)

---

## What is completed

### Task 3.1 — Clean KB Curation
- Implemented in `scripts/build_kb.py`.
- Produces `data/clean_kb/kb.jsonl` with 5,000 docs.
- Includes schema-aware NQ title extraction, Wikipedia fetch/chunking, one-hop expansion, and dedup.

### Task 3.2 — Vector Store Build
- Implemented in `scripts/build_vectorstore.py`.
- Builds persistent Chroma store at `data/vectorstore/clean`.
- Collection compatibility handled (`kb` logical name, legacy physical `kb0` fallback).

### Task 3.3 — Clean RAG Pipeline
- Implemented in `scripts/build_rag_pipeline.py`.
- Retrieval path is stable and unchanged from clean store flow.
- Generation backend supports:
  - `huggingface`
  - `ollama` (validated locally with `llama3.1:8b`)
- Prompt was tightened to produce short answer spans and return exactly `unknown` when context lacks the answer.

### Task 3.4 — Clean Evaluation
- Implemented in `scripts/evaluate_rag.py`.
- Loads first N examples from Natural Questions (default 1000).
- Runs clean RAG pipeline per question.
- Computes and logs:
  - Exact Match (EM)
  - token-level F1
- Appends metrics to `results/metrics.csv`.
- Baseline metadata fixed as required:
  - `attack_type=clean`
  - `poison_rate=0.0`
  - `detector=none`

---

## Ready files / artifacts

### Core scripts
- `scripts/build_kb.py`
- `scripts/build_vectorstore.py`
- `scripts/build_rag_pipeline.py`
- `scripts/evaluate_rag.py`

### Data / index artifacts (generated)
- `data/clean_kb/kb.jsonl` (current local size: ~9.3 MB)
- `data/vectorstore/clean/` (current local size: ~92 MB)

### Evaluation outputs
- `results/metrics.csv`
- `results/metrics_prompt_tuning.csv`
- `results/predictions_clean_smoke.jsonl`
- `results/predictions_clean_20_old.jsonl`
- `results/predictions_clean_20_new.jsonl`
- `results/predictions_clean_100.jsonl`
- `results/predictions_clean.jsonl`

---

## What is still provisional
- Evaluation quality is **provisional baseline quality**, not final tuned performance.
- Latest verified prompt-tuning comparison on 20 examples:
  - old: `EM=0.0000`, `token_F1=0.1022`
  - new: `EM=0.0500`, `token_F1=0.1091`
- This indicates improvement, but metrics are still modest and should be treated as baseline.
- Full-scale reporting should be based on a larger run (e.g., 100 then 1000 with final frozen prompt/settings).

---

## How teammates should use these outputs
1. Use `scripts/build_rag_pipeline.py` for clean RAG inference (local Ollama path recommended for immediate local reproducibility).
2. Use `scripts/evaluate_rag.py` for baseline EM/F1 evaluation over NQ.
3. Keep retrieval/vector store flow unchanged while comparing later modules.
4. Treat current results as the clean baseline checkpoint before poisoning/detection experiments.

---

## Important caveats
- Chroma telemetry warning messages may appear during runs; current pipeline remains runtime-stable.
- Collection resolution may map requested `kb` to physical `kb0` depending on persisted store contents.
- `data/vectorstore/clean` is a large binary artifact and not ideal for normal Git history.
- Evaluation outputs are model/prompt sensitive; current numbers are baseline, not final target quality.
