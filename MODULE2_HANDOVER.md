# Module 2 — Detection & Defense Layer · Handoff

Hey team! Module 2 handoff is now pushed to GitHub and ready for use.

**Repo:** https://github.com/vpate160/KRR-Final-Project
**Branch:** `main` (PR #3 merged)
**Owner:** Anushree Bhure

## What's implemented

- **Task 2.1** — embedding extraction with `sentence-transformers/all-mpnet-base-v2` over clean KB + all 5 poisoned variants
- **Task 2.2** — Isolation Forest + LOF unsupervised detectors trained on clean embeddings, scored on each variant
- **Task 2.3** — 2-layer MLP supervised classifier (768 → 256 → 64 → 1) with per-variant training and validation-tuned thresholds
- **Task 2.4a** — GPT-2 perplexity baseline (top-K% threshold = expected poison rate)
- **Task 2.4b** — Llama 3.1 8B LLM-as-judge baseline via local Ollama (4 of 5 variants)
- **Task 2.5** — filter-based mitigation analysis across all 5 variants

## Current artifacts available

- per-variant embeddings — `Module 2/data/embeddings/`
- per-doc detection scores for all 5 detectors — `Module 2/results/scores/`
- per-variant MLP checkpoints with tuned thresholds — `Module 2/models/`
- detection metrics CSV (24 rows) — `Module 2/results/detection_metrics.csv`
- mitigation filter summary CSV (5 rows) — `Module 2/results/mitigation_filter_summary.csv`
- filtered KBs per variant — `Module 2/data/filtered_kb/`
- SLURM job script for SOL — `Module 2/scripts/run_module2_sol.sh`

## Important note

The headline finding is a **negative result** that the report should foreground: every embedding-space detector is at or near random (ROC-AUC 0.47 – 0.60) on Module 1's stealthy attacks. Direct measurement: cosine(poisoned, clean original) = **0.9958** mean on `factual_0.1`. mpnet is essentially invariant to single-fact swaps, so the project's central hypothesis is falsified for this attack family. The filter analysis confirms downstream — filter precision tracks the poison rate exactly, which is the signature of random selection.

Task 2.5's RAG re-evaluation on Natural Questions was not run end-to-end. NQ first-time download via `datasets.load_dataset` triggers a full 287-shard pull (~30 min on M2) that exceeded today's submission window. The filter-only analysis already establishes the conclusion; the EM/F1 recovery row is a clean follow-up to run on SOL once accessible.

## Implementation note

The full evaluation was run locally on Apple M2 with MPS rather than on SOL, since SOL access was unavailable today. Llama 3.1 8B for the LLM-judge baseline runs through local Ollama (`llama3.1:8b` Q4_K_M) — same backend Vatsal is using for Module 3, so embeddings and judge outputs are comparable.

## Integration status

- **Hardik** — no further input needed from your side; your `data/poisoned_kb/` exports were consumed end-to-end.
- **Latika** — clean + per-variant embeddings, per-doc scores for all 5 detectors, and both metrics CSVs are on `main`. UMAP plots will show poisoned and clean overlapping (the qualitative version of AUC ≈ 0.5) — that overlap is itself a useful figure for the report.
- **Lance** — `detection_metrics.csv` + `mitigation_filter_summary.csv` are the source of truth for the report's Tables 1–3. Headline numbers and the falsification framing are captured in PR #3's description if helpful for the methods/analysis sections.
- **Vatsal** — thanks for the merge; Module 2 is on `main`. No further changes needed from M3 unless you want to re-run Task 2.5's NQ eval on SOL once HF access is sorted.

Please pull the latest `main` and use `Module 2/README.md` + the PR #3 description as the starting point.
