# Final Visualization Package

## Narrative To Use

Frame the result as the **Stealthy Poisoning Paradox**:

1. **Overlap finding:** high-quality factual swaps and semantic distortions do not create a clean anomaly cluster in embedding space. Detector AUCs average **0.527**, close to random guessing.
2. **Naive mitigation crisis:** the old threshold strategy can remove too much clean evidence. `factual_0.2` is the clearest warning case because threshold filtering emptied the KB.
3. **Safe mitigation result:** top-K capped mitigation now covers all five variants and preserves KB integrity. It is stable, but it is neutral rather than performance-improving because token-F1 matches undefended retrieval at about **0.0690**.

## Visual Assets

Use these files directly from `results/figures/`:

| File | Use in report |
|---|---|
| `results/figures/umap_small_multiples.png` | Five-variant embedding overlap figure. Filename is report-friendly; local projection is t-SNE because `umap-learn` is not installed. |
| `results/figures/rag_accuracy_curves.png` | RAG strategy figure comparing undefended, naive threshold, and top-K capped mitigation. |
| `results/figures/detector_comparison_bar.png` | AUC comparison figure with the random-guess line at 0.5. |

Supporting tables:

| Table | Path |
|---|---|
| RAG accuracy/recovery | `results/tables/module4_table1_rag_accuracy.csv` |
| Detector performance | `results/tables/module4_table2_detector_performance.csv` |
| Mitigation summary | `results/tables/module4_table3_mitigation_filter_summary.csv` |

## Failure Taxonomy Categories

- **Category 1: Hard Factual Swaps:** one date/name/fact changes while the surrounding passage stays almost identical, causing minimal embedding shift.
- **Category 2: Semantic Overlap:** the rewrite is fluent and topical, so all-mpnet-base-v2 treats it like a natural paraphrase instead of a threat.
- **Category 3: Boundary Cases:** scores fall near the practical cutoff, so threshold changes either miss poison or remove too much clean evidence.

Detailed before/after examples from Hardik's change logs are in `results/failure_analysis/failure_taxonomy_draft.md`.

## Final Technical Check

- Report PNG assets are generated at 200 DPI.
- EM/F1 normalization is centralized in `scripts/evaluate_rag.py`: lowercase, punctuation removal, article removal, and whitespace normalization before scoring.
- Threshold `factual_0.2` is not a missing visualization; it is a real mitigation failure caused by an empty filtered KB. Top-K capped `factual_0.2` is now present.

## Contribution Table

| Contributor | Contribution |
|---|---|
| Latika | Final visualization dashboard, embedding-space overlap analysis, detector/mitigation/RAG performance plots, error analysis, and failure taxonomy. |
| Anushree | Detection metrics, embeddings, detector scores, mitigation summary, and top-K capped mitigation rows. |
| Hardik | Poisoned KB variants, labels, and change logs used for text-backed taxonomy examples. |
| Vatsal | RAG baseline, undefended, threshold-mitigated, and top-K capped recovery metrics. |
| Lance | Final report integration and narrative synthesis. |
