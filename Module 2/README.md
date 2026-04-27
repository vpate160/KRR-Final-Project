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
- Clean KB: `../data/clean_kb/kb.jsonl` (Vatsal — already in repo).
- Clean embeddings: `../data/embeddings/clean_embeddings.npy` + `clean_doc_ids.json` (Vatsal — already in repo).
- Poisoned KBs: `../data/poisoned_kb/<variant>.jsonl` plus `<variant>_labels.npy` per variant (Hardik). Module 1 ships `factual_0.1`, `factual_0.2`, `factual_0.3`, `semantic_0.1`, `semantic_0.3`. The detection code auto-discovers variants from `../data/poisoned_kb/` (Module 1's location) and falls back to `Module 2/data/poisoned_kb/` for legacy `poisoned_<variant>.jsonl` exports.

Pulling Module 1's data into your working tree:

```bash
# from the repo root, one-time setup
git remote add hardik https://github.com/hpareek871/KRR-Final-Project.git
git fetch hardik

# bring in the latest poisoned KB variants and labels
git checkout hardik/main -- data/poisoned_kb/ logs/
```

Typical run order (from the `Module 2/` directory):

```bash
# Task 2.1 — encode poisoned KBs; clean embeddings are reused from Module 3
python -m scripts.run_2_1_extract                          # all 5 variants
python -m scripts.run_2_1_extract --variants factual_0.1   # smoke

# Task 2.2 — unsupervised detectors (Isolation Forest + LOF)
python -m scripts.run_2_2_anomaly

# Task 2.3 — supervised MLP
python -m scripts.run_2_3_neural

# Task 2.4 — baselines
python -m scripts.run_2_4_perplexity
python -m scripts.run_2_4_llm_judge      # requires Ollama running locally

# Task 2.5 — mitigation (rebuilds vector store + calls Module 3's evaluator)
python -m scripts.run_2_5_mitigate --variant factual_0.1 --run-undefended
```

## Running on SOL

The bulk of the pipeline (Tasks 2.1, 2.2, 2.3, 2.4 perplexity) is bundled in
`scripts/run_module2_sol.sh`, a SLURM script modelled on Vatsal's
`scripts/run_rag_query_sol.sh`. Submit it from the repo root:

```bash
ssh <asurite>@sol.asu.edu
cd /scratch/<asurite>/KRR-Final-Project
git pull origin anushree/module-2-detection
git fetch hardik && git checkout hardik/main -- data/poisoned_kb/ logs/
sbatch "Module 2/scripts/run_module2_sol.sh"
```

Tasks 2.4 LLM-judge and 2.5 require an Ollama server reachable at
`http://localhost:11434`, so they are not part of the default SLURM job. Run
them on a node where Ollama is already serving, or start Ollama inside an
allocated job before invoking those scripts.

## Respecting Module 3's guardrails

- Retrieval and prompt settings in `../scripts/build_rag_pipeline.py` and
  `../scripts/evaluate_rag.py` are **frozen** for baseline comparability.
  Task 2.5 changes the *input corpus* only — never the retrieval code.
- Module 3's collection-name resolution (`kb` → `kb_store` → `kb0`) is reused as-is.
