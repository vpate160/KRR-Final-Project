from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import datasets as _datasets

_M2 = Path(__file__).resolve().parents[1]
if str(_M2) not in sys.path:
    sys.path.insert(0, str(_M2))

_orig = _datasets.load_dataset
def _patched(*args, **kwargs):
    name = args[0] if args else kwargs.get("path")
    split = kwargs.get("split", "")
    if name == "natural_questions" and isinstance(split, str) and "[:" in split:
        n = int(split.split("[:")[1].rstrip("]"))
        ds = _orig(name, split=split.split("[:")[0], streaming=True)
        out = []
        it = iter(ds)
        for _ in range(n):
            try: out.append(next(it))
            except StopIteration: break
        return out
    return _orig(*args, **kwargs)
_datasets.load_dataset = _patched

from src.detection.mitigation import (
    _predict_neural, filter_kb, build_vectorstore_subprocess,
    run_module3_evaluator, append_module3_metrics, _override_row,
    MITIGATED_DETECTOR_TAG,
)
from src.detection.utils import (
    setup_logging, variant_paths, M2_FILTERED_KB, M2_VECTORSTORES,
)

LOGGER = logging.getLogger("run_2_5_topk")

VARIANTS = [
    ("factual_0.1", 0.1),
    ("factual_0.2", 0.2),
    ("factual_0.3", 0.3),
    ("semantic_0.1", 0.1),
    ("semantic_0.3", 0.3),
]

def main() -> int:
    setup_logging("INFO")
    n_examples = 5
    for variant, poison_rate in VARIANTS:
        LOGGER.info("=== top-K mitigation variant=%s K=%.2f ===", variant, poison_rate)
        import torch
        from src.detection.utils import M2_MODELS
        from src.detection.neural_classifier import _make_model
        from src.detection.embeddings import load_variant
        embeddings, doc_ids, _ = load_variant(variant)
        ckpt = torch.load(M2_MODELS / f"{variant}_mlp.pt", map_location="cpu", weights_only=False)
        model = _make_model()
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.from_numpy(np.ascontiguousarray(embeddings)).float())).numpy().ravel()
        k = int(round(poison_rate * len(doc_ids)))
        top_idx = np.argsort(-probs)[:k]
        flagged = set(doc_ids[i] for i in top_idx)
        LOGGER.info("Flagged top-%d/%d docs", k, len(doc_ids))

        kb_in = variant_paths(variant)["kb"]
        kb_out = M2_FILTERED_KB / f"filtered_{variant}_topK.jsonl"
        filter_kb(kb_in, flagged, kb_out)

        vs_path = M2_VECTORSTORES / f"filtered_{variant}_topK"
        build_vectorstore_subprocess(kb_out, vs_path, "kb")

        metrics = run_module3_evaluator(
            vectorstore_path=vs_path,
            collection_name="kb",
            n_examples=n_examples,
        )
        row = _override_row(metrics, variant=variant, detector_tag="mitigated_topK")
        append_module3_metrics(row)
        em = metrics.get("exact_match", 0.0)
        f1 = metrics.get("token_f1", 0.0)
        print(f"[{variant} top-K%] kept={1000-len(flagged)} dropped={len(flagged)} EM={em:.4f} F1={f1:.4f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
