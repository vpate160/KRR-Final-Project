"""Task 2.5 — Mitigation pipeline.

For a chosen detector + poisoned variant:
1. Flag poisoned documents (re-scoring the variant with the chosen detector).
2. Produce a filtered KB jsonl with flagged docs removed.
3. Rebuild a Chroma vector store over the filtered KB by invoking Module 3's
   unchanged `scripts/build_vectorstore.py` as a subprocess.
4. Re-run Module 3's unchanged evaluator (`evaluate_clean_rag_on_nq`) against
   the filtered store.
5. Append a recovery row to the repo-root `results/metrics.csv` using Module 3's
   exact schema, with `detector=mitigated`.

Optionally runs the "undefended" (poisoned, unfiltered) variant too, for a
same-run before/after comparison. That phase produces Hardik's degradation row
if he hasn't already.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .anomaly_detector import (
    ISOLATION_FOREST,
    LOF,
    _attack_type_from_variant,
    _poison_rate_from_variant,
    train_detectors,
)
from .embeddings import load_variant
from .neural_classifier import NEURAL, _make_model
from .utils import (
    EMBEDDING_MODEL,
    M2_FILTERED_KB,
    M2_MODELS,
    M2_VECTORSTORES,
    MODULE3_SCRIPTS,
    REPO_ROOT,
    ROOT_METRICS_CSV,
    SEED,
    load_jsonl,
    save_jsonl,
    set_seed,
    variant_paths,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_TOP_K = 5
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.1

MITIGATED_DETECTOR_TAG = "mitigated"
UNDEFENDED_DETECTOR_TAG = "none"


@dataclass
class MitigationOutcome:
    variant: str
    detector: str
    n_flagged: int
    n_kept: int
    filtered_kb_path: Path
    filtered_vectorstore_path: Path
    defended_metrics: Dict[str, Any]
    undefended_metrics: Optional[Dict[str, Any]] = None
    flagged_doc_ids: List[str] = field(default_factory=list)


def _predict_isoforest_or_lof(detector_name: str, variant: str) -> Tuple[List[str], np.ndarray]:
    embeddings, doc_ids, _ = load_variant(variant)
    clean_emb, _, _ = load_variant("clean")
    detectors = train_detectors(clean_emb)
    det = detectors[detector_name]
    preds = (det.predict(embeddings) == -1).astype(int)  # type: ignore[attr-defined]
    return doc_ids, preds


def _predict_neural(variant: str, threshold_override: Optional[float]) -> Tuple[List[str], np.ndarray]:
    import torch

    ckpt_path = M2_MODELS / f"{variant}_mlp.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No trained MLP for variant '{variant}' at {ckpt_path}. Run Task 2.3 first."
        )
    embeddings, doc_ids, _ = load_variant(variant)

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    model = _make_model()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    threshold = threshold_override if threshold_override is not None else float(ckpt["threshold"])
    LOGGER.info("Scoring %d docs with trained MLP (threshold=%.2f)", len(doc_ids), threshold)

    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(embeddings)).to(torch.float32)
        probs = torch.sigmoid(model(x)).numpy().ravel()
    preds = (probs >= threshold).astype(int)
    return doc_ids, preds


def flag_documents(
    variant: str,
    detector: str,
    threshold_override: Optional[float] = None,
) -> Tuple[List[str], np.ndarray]:
    """Return (doc_ids, binary_preds) where preds[i]=1 means "flag this doc"."""
    set_seed()
    if detector in (ISOLATION_FOREST, LOF):
        return _predict_isoforest_or_lof(detector, variant)
    if detector == NEURAL:
        return _predict_neural(variant, threshold_override)
    raise ValueError(f"Unknown detector for mitigation: {detector}")


def filter_kb(
    src_kb_path: Path,
    flagged_ids: Set[str],
    out_path: Path,
) -> Tuple[int, int]:
    """Write a filtered copy of `src_kb_path` with docs in `flagged_ids` removed."""
    if not src_kb_path.exists():
        raise FileNotFoundError(f"KB not found: {src_kb_path}")
    records = load_jsonl(src_kb_path)
    kept = [r for r in records if r["id"] not in flagged_ids]
    save_jsonl(out_path, kept)
    n_flagged = len(records) - len(kept)
    LOGGER.info(
        "Filtered KB: %d -> %d docs (dropped %d) -> %s",
        len(records),
        len(kept),
        n_flagged,
        out_path,
    )
    return n_flagged, len(kept)


def _clean_vectorstore_dir(path: Path) -> None:
    if path.exists():
        LOGGER.info("Removing stale vectorstore dir: %s", path)
        shutil.rmtree(path)


def build_vectorstore_subprocess(
    kb_path: Path,
    persist_dir: Path,
    collection_name: str = "kb",
    embedding_model: str = EMBEDDING_MODEL,
) -> None:
    """Invoke Vatsal's unchanged build_vectorstore.py as a subprocess."""
    _clean_vectorstore_dir(persist_dir)
    cmd = [
        sys.executable,
        str(MODULE3_SCRIPTS / "build_vectorstore.py"),
        "--kb-path",
        str(kb_path),
        "--persist-dir",
        str(persist_dir),
        "--collection-name",
        collection_name,
        "--embedding-model",
        embedding_model,
    ]
    LOGGER.info("Building vector store: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def run_module3_evaluator(
    vectorstore_path: Path,
    *,
    n_examples: int,
    collection_name: str = "kb",
    generation_backend: str = "ollama",
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_base_url: str = DEFAULT_OLLAMA_URL,
    top_k: int = DEFAULT_TOP_K,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    predictions_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Call Module 3's unchanged evaluator and return the raw metrics dict."""
    if str(MODULE3_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(MODULE3_SCRIPTS))
    from evaluate_rag import evaluate_clean_rag_on_nq  # type: ignore

    LOGGER.info(
        "Running Module 3 evaluator on vectorstore=%s with n_examples=%d", vectorstore_path, n_examples
    )
    if predictions_path is not None and predictions_path.exists():
        predictions_path.unlink()
    metrics = evaluate_clean_rag_on_nq(
        vectorstore_path=str(vectorstore_path),
        collection_name=collection_name,
        embedding_model=EMBEDDING_MODEL,
        generation_backend=generation_backend,
        model_path="meta-llama/Meta-Llama-3.1-8B-Instruct",
        ollama_model=ollama_model,
        ollama_base_url=ollama_base_url,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        n_examples=n_examples,
        predictions_path=predictions_path,
    )
    LOGGER.info("Eval done: EM=%.4f F1=%.4f", metrics.get("exact_match", 0.0), metrics.get("token_f1", 0.0))
    return dict(metrics)


def append_module3_metrics(row: Dict[str, Any]) -> None:
    """Append a row to the repo-root results/metrics.csv using Module 3's writer."""
    if str(MODULE3_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(MODULE3_SCRIPTS))
    from evaluate_rag import append_metrics_row  # type: ignore

    ROOT_METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    append_metrics_row(ROOT_METRICS_CSV, row)


def _override_row(
    base_metrics: Dict[str, Any],
    *,
    variant: str,
    detector_tag: str,
) -> Dict[str, Any]:
    row = dict(base_metrics)
    row["attack_type"] = _attack_type_from_variant(variant)
    row["poison_rate"] = _poison_rate_from_variant(variant)
    row["detector"] = detector_tag
    row["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    return row


def run_mitigation(
    variant: str,
    detector: str,
    *,
    n_examples: int = 1000,
    threshold: Optional[float] = None,
    run_undefended: bool = False,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_base_url: str = DEFAULT_OLLAMA_URL,
    predictions_root: Optional[Path] = None,
) -> MitigationOutcome:
    if variant == "clean":
        raise ValueError("Cannot mitigate the 'clean' variant (nothing to filter).")

    # 1. Flag documents.
    doc_ids, preds = flag_documents(variant, detector, threshold_override=threshold)
    flagged_ids = {doc_ids[i] for i, flag in enumerate(preds) if flag == 1}
    LOGGER.info(
        "Detector '%s' flagged %d/%d docs in variant '%s'",
        detector,
        len(flagged_ids),
        len(doc_ids),
        variant,
    )

    # 2. Write filtered KB.
    poisoned_kb_path = variant_paths(variant)["kb"]
    if not poisoned_kb_path.exists():
        raise FileNotFoundError(
            f"Poisoned KB not found for variant '{variant}' at {poisoned_kb_path}. "
            "Ensure Hardik's file is staged under Module 2/data/poisoned_kb/."
        )
    filtered_kb_path = M2_FILTERED_KB / f"filtered_{variant}_{detector}.jsonl"
    n_flagged, n_kept = filter_kb(poisoned_kb_path, flagged_ids, filtered_kb_path)

    # 3. Rebuild vector store over filtered KB.
    defended_vs = M2_VECTORSTORES / f"filtered_{variant}_{detector}"
    build_vectorstore_subprocess(filtered_kb_path, defended_vs)

    # 4. Run evaluator on filtered store.
    def _pred_path(tag: str) -> Optional[Path]:
        if predictions_root is None:
            return None
        predictions_root.mkdir(parents=True, exist_ok=True)
        return predictions_root / f"predictions_{variant}_{tag}.jsonl"

    defended_metrics = run_module3_evaluator(
        defended_vs,
        n_examples=n_examples,
        ollama_model=ollama_model,
        ollama_base_url=ollama_base_url,
        predictions_path=_pred_path(f"defended_{detector}"),
    )
    defended_row = _override_row(
        defended_metrics,
        variant=variant,
        detector_tag=MITIGATED_DETECTOR_TAG,
    )
    append_module3_metrics(defended_row)

    undefended_metrics: Optional[Dict[str, Any]] = None
    if run_undefended:
        undefended_vs = M2_VECTORSTORES / f"poisoned_{variant}"
        build_vectorstore_subprocess(poisoned_kb_path, undefended_vs)
        undefended_metrics = run_module3_evaluator(
            undefended_vs,
            n_examples=n_examples,
            ollama_model=ollama_model,
            ollama_base_url=ollama_base_url,
            predictions_path=_pred_path("undefended"),
        )
        undef_row = _override_row(
            undefended_metrics,
            variant=variant,
            detector_tag=UNDEFENDED_DETECTOR_TAG,
        )
        append_module3_metrics(undef_row)

    return MitigationOutcome(
        variant=variant,
        detector=detector,
        n_flagged=n_flagged,
        n_kept=n_kept,
        filtered_kb_path=filtered_kb_path,
        filtered_vectorstore_path=defended_vs,
        defended_metrics=defended_row,
        undefended_metrics=undef_row if run_undefended else None,
        flagged_doc_ids=sorted(flagged_ids),
    )
