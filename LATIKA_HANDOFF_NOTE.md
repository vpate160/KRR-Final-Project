# Latika Handoff — Mitigation Metrics Status (Apr 27, 2026)

## Completed variants (both undefended + mitigated rows now present)
- factual_0.1
- factual_0.3
- semantic_0.1
- semantic_0.3

These rows are appended to root `results/metrics.csv` using the Module 2 mitigation pipeline (neural classifier, threshold=0.5, n=20, `--run-undefended`).

## Missing variant
- **factual_0.2** is still missing (both undefended + mitigated rows).

## Why factual_0.2 failed
The MLP detector flagged **100%** of docs for `factual_0.2` at threshold **0.5**, producing an empty filtered KB. `build_vectorstore.py` fails on empty KB (`No valid records found in KB JSONL`), so the mitigation run aborts before any metrics rows can be appended.

## Guidance for plots
You can proceed with plots and tables for the completed variants listed above. Treat `factual_0.2` as pending until we agree on a detector/threshold adjustment.
