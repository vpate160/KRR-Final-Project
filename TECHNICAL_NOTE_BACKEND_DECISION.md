# Technical Note: Local Backend Decision (Mac) — April 27, 2026

## Decision Summary

For this Mac local environment:

1. **Hugging Face backend support remains enabled in code.**
2. **Hugging Face is considered validated for single-query use only** (interactive/local sanity queries).
3. **Ollama is the practical backend for local evaluation runs** (multi-example/benchmark loops).
4. **No additional local Hugging Face multi-example evaluations should be launched** at this time.

## Scope Boundaries

- No changes to KB logic.
- No changes to vector store logic.
- No further changes to evaluation logic in this step.

## Rationale (Operational)

- Local HF single-query execution is useful for functional validation.
- Multi-example local evaluation is operationally more stable/practical with Ollama on this machine.
- This keeps progress predictable while preserving HF support for future environments.

## Immediate Team Guidance

- Use `scripts/build_rag_pipeline.py` with HF backend for targeted single-query checks when needed.
- Use `scripts/evaluate_rag.py` with Ollama backend for local smoke/benchmark evaluations.
