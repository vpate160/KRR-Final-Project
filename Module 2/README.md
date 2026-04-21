# Module 2 — Detection & Defense Layer

Owner: **Anushree Bhure**

This folder contains all code and outputs for Module 2 of Project 07 (CSE 579 KRR).
The rest of the repository (Module 3, owned by Vatsal) is treated as **read-only** here.

## What Module 2 does

Given a knowledge base that may contain poisoned documents (produced by Hardik's Module 1),
this module:

1. Extracts 768-dim `all-mpnet-base-v2` embeddings for each document. (Task 2.1)
2. Flags poisoned documents using two detectors: Isolation Forest + LOF (unsupervised,
   Task 2.2) and a 2-layer MLP binary classifier (supervised, Task 2.3).
3. Compares the detectors against two baselines: GPT-2 perplexity and LLM-as-judge
   using the same local `llama3.1:8b` via Ollama that Module 3 uses. (Task 2.4)
4. Builds a filtered knowledge base from the flagged documents, rebuilds a Chroma
   vector store using Module 3's unchanged `scripts/build_vectorstore.py`, and re-runs
   Module 3's unchanged `scripts/evaluate_rag.py` to measure recovery. (Task 2.5)

## Directory layout

```
Module 2/
├── src/detection/        # library code (importable)
├── scripts/              # thin CLI entrypoints for each task
├── data/
│   ├── embeddings/       # .npy files per KB variant (gitignored, large)
│   ├── poisoned_kb/      # staging copy of Hardik's poisoned jsonl files (gitignored)
│   ├── filtered_kb/      # defended jsonl files produced by Task 2.5 (gitignored)
│   └── vectorstores/     # Chroma stores over filtered KBs (gitignored, large)
├── models/               # trained MLP checkpoints (gitignored)
├── results/
│   ├── detection_metrics.csv   # precision/recall/F1/AUC per detector (this file IS committed)
│   └── scores/                 # per-doc detector scores .npy (gitignored, large)
└── notebooks/            # exploratory / sanity-check notebooks
```

## Two CSVs, intentionally

- `results/metrics.csv` (at the repo root, owned by Vatsal/Latika): mitigation recovery
  runs append rows here using Module 3's existing schema with `detector=mitigated`.
  Module 2 never changes this file's column set.
- `Module 2/results/detection_metrics.csv` (owned by Module 2): pure detection runs log
  `precision / recall / F1 / AUC` here. This schema has no EM/F1 because detection
  experiments don't run the RAG pipeline.

## How to run (once inputs are ready)

Upstream dependencies:
- Clean KB: `../data/clean_kb/kb.jsonl` (Vatsal, already in repo)
- Clean embeddings: `../data/embeddings/clean_embeddings.npy` + `clean_doc_ids.json` (Vatsal, already in repo)
- Poisoned KBs: `data/poisoned_kb/poisoned_factual_*.jsonl` etc. (Hardik, copy here when ready)

Typical run order:

```bash
# Task 2.1 — encode poisoned KBs; clean embeddings are reused from Module 3
python -m scripts.run_2_1_extract

# Task 2.2 — unsupervised detectors
python -m scripts.run_2_2_anomaly

# Task 2.3 — supervised MLP
python -m scripts.run_2_3_neural

# Task 2.4 — baselines
python -m scripts.run_2_4_perplexity
python -m scripts.run_2_4_llm_judge      # requires Ollama running locally

# Task 2.5 — mitigation (rebuilds vector store + calls Module 3's evaluator)
python -m scripts.run_2_5_mitigate
```

## Respecting Module 3's guardrails

- Retrieval and prompt settings in `../scripts/build_rag_pipeline.py` and
  `../scripts/evaluate_rag.py` are **frozen** for baseline comparability.
  Task 2.5 changes the *input corpus* only — never the retrieval code.
- Module 3's collection-name resolution (`kb` → `kb_store` → `kb0`) is reused as-is.
