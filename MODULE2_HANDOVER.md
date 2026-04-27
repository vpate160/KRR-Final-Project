# Module 2 — Detection & Defense · Handoff

**Repo:** vpate160/KRR-Final-Project · **Branch:** `main` (PR #3 merged) · **Owner:** Anushree Bhure

## Tasks shipped

| Task | Detector | Implementation | Output |
|---|---|---|---|
| 2.1 | mpnet embeddings | `sentence-transformers/all-mpnet-base-v2`, MPS, batch=64, seed=42 | `Module 2/data/embeddings/<variant>_{embeddings.npy,doc_ids.json,labels.npy}` |
| 2.2 | IsolationForest, LOF | n_estimators=200 / n_neighbors=20, fit on 5000 clean, score 1000-doc variant | `Module 2/results/scores/<variant>_{isolation_forest,lof}_scores.npy` |
| 2.3 | 2-layer MLP | 768→256→64→1, BCE + early stop, 70/15/15 stratified split, val-tuned threshold | `Module 2/models/<variant>_mlp.pt` (state_dict + threshold) |
| 2.4a | GPT-2 perplexity | sliding NLL (max_len=1024, stride=512), top-K%% threshold = expected poison rate | `Module 2/results/scores/<variant>_perplexity_scores.npy` |
| 2.4b | Llama 3.1 8B judge | Q4_K_M via local Ollama, stratified n=100/variant, yes/no parse | `Module 2/results/scores/<variant>_llm_judge_scores.npy` (4/5) |
| 2.5 | Filter (neural, t=0.5) | MLP rescoring on full 1000 docs, drop flagged | `Module 2/data/filtered_kb/filtered_<variant>_neural_classifier.jsonl` |

Variants: `factual_0.1`, `factual_0.2`, `factual_0.3`, `semantic_0.1`, `semantic_0.3` (1000 docs each, labels via `<variant>_labels.npy`).

## Aggregate results

`Module 2/results/detection_metrics.csv` — 24 rows (5 IF + 5 LOF + 5 MLP + 5 perplexity + 4 LLM-judge).
`Module 2/results/mitigation_filter_summary.csv` — 5 rows.

```
Detector             ROC-AUC range    Best F1
isolation_forest     0.47 – 0.55      0.000  (predict() collapsed under contamination='auto')
lof                  0.50 – 0.55      0.000  (same)
neural_classifier    0.50 – 0.58      0.466  (recall-collapse at val threshold 0.05)
perplexity           0.51 – 0.60      0.300  (top-K% with K = poison rate)
llm_judge            n/a (binary)     0.154  (model answers "no" on >95% of inputs)
```

Diagnostic: cos(emb(poisoned), emb(clean_original)) = **0.9958 mean**, 0.9659 min over a 30-pair sample on `factual_0.1`. mpnet is invariant to Module 1's single-fact swaps; embedding-space hypothesis is **falsified** for this attack family.

Filter precision per variant tracks the ground-truth poison rate to two decimals — random selection.

## Task 2.5 RAG re-evaluation (NQ via streaming)

`datasets.load_dataset("natural_questions", split="train[:N]")` is monkey-patched to streaming mode in `Module 2/scripts/_run_2_5_with_streaming_nq.py` to bypass the full 287-shard download. n_examples=5 per variant. Rows in root `results/metrics.csv` (`detector=mitigated`):

```
variant         defended EM  defended F1   undefended F1
factual_0.1     0.000        0.0023        0.0690
factual_0.3     0.000        0.0423        --
semantic_0.1    0.000        0.0023        --
semantic_0.3    0.000        0.0023        --
factual_0.2     skipped (filtered KB has 0 docs)
```

Defended F1 collapses below the undefended baseline because the filter destroys the KB — confirming the detection-at-chance ⇒ mitigation-fails prediction.

## Backend

Local M2 + MPS. Llama 3.1 8B Q4_K_M via Ollama at `http://localhost:11434` (matches Vatsal's M3 backend; embeddings and judge outputs are comparable across modules).

## Reproduce

```bash
git pull origin main
git remote add hardik https://github.com/hpareek871/KRR-Final-Project.git
git fetch hardik && git checkout hardik/main -- data/poisoned_kb/ logs/
source .venv/bin/activate
cd "Module 2"
python scripts/run_2_1_extract.py
python scripts/run_2_2_anomaly.py
python scripts/run_2_3_neural.py
python scripts/run_2_4_perplexity.py
python scripts/run_2_4_llm_judge.py --max-docs 100
```

SOL: `sbatch "Module 2/scripts/run_module2_sol.sh"` from repo root.

## Consumers

- **M4 (Latika):** `Module 2/data/embeddings/`, `Module 2/results/scores/`, both CSVs.
- **M5 (Lance):** both CSVs + the cosine-invariance number for the falsification framing.
- **M1 (Hardik) / M3 (Vatsal):** no further input required.
