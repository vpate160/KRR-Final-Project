from __future__ import annotations

import logging
import sys
from pathlib import Path

import datasets as _datasets

_MODULE2_ROOT = Path(__file__).resolve().parents[1]
if str(_MODULE2_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE2_ROOT))

_orig_load_dataset = _datasets.load_dataset

def _streaming_load_dataset(*args, **kwargs):
    name = args[0] if args else kwargs.get("path")
    split = kwargs.get("split", "")
    if name == "natural_questions" and isinstance(split, str) and "[:" in split:
        n = int(split.split("[:")[1].rstrip("]"))
        base_split = split.split("[:")[0]
        ds = _orig_load_dataset(name, split=base_split, streaming=True)
        materialized = []
        it = iter(ds)
        for _ in range(n):
            try:
                materialized.append(next(it))
            except StopIteration:
                break
        return materialized
    return _orig_load_dataset(*args, **kwargs)

_datasets.load_dataset = _streaming_load_dataset

from src.detection.mitigation import run_mitigation
from src.detection.utils import setup_logging

LOGGER = logging.getLogger("run_2_5_streaming")

def main() -> int:
    setup_logging("INFO")
    variants = ["factual_0.1", "factual_0.2", "factual_0.3", "semantic_0.1", "semantic_0.3"]
    failures: list[str] = []
    for i, variant in enumerate(variants):
        run_undefended = (i == 0)
        LOGGER.info("=== mitigation variant=%s undefended=%s ===", variant, run_undefended)
        try:
            outcome = run_mitigation(
                variant=variant,
                detector="neural_classifier",
                n_examples=5,
                threshold=0.5,
                run_undefended=run_undefended,
            )
            print(f"[{variant}] flagged={outcome.n_flagged} kept={outcome.n_kept}")
            print(f"  defended  EM={outcome.defended_metrics.get('exact_match',0):.4f}  F1={outcome.defended_metrics.get('token_f1',0):.4f}")
            if outcome.undefended_metrics:
                print(f"  undefended EM={outcome.undefended_metrics.get('exact_match',0):.4f}  F1={outcome.undefended_metrics.get('token_f1',0):.4f}")
        except Exception as exc:
            LOGGER.exception("variant %s failed: %s", variant, exc)
            failures.append(variant)
    if failures:
        LOGGER.error("Failures: %s", failures)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
