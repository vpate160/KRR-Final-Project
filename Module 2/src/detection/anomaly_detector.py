from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import LocalOutlierFactor

from .embeddings import load_variant
from .utils import (
    M2_SCORES,
    SEED,
    append_detection_row,
    set_seed,
)

LOGGER = logging.getLogger(__name__)

ISOLATION_FOREST = "isolation_forest"
LOF = "lof"

@dataclass
class DetectorResult:
    detector: str
    variant: str
    scores: np.ndarray
    preds: np.ndarray
    labels: np.ndarray
    precision: float
    recall: float
    f1: float
    roc_auc: float
    threshold_desc: str = "auto"

def _poison_rate_from_variant(variant: str) -> float:
    for part in variant.split("_"):
        if part.endswith("pct"):
            try:
                return int(part[:-3]) / 100.0
            except ValueError:
                return 0.0
    return 0.0

def _attack_type_from_variant(variant: str) -> str:
    if variant.startswith("factual"):
        return "factual_swap"
    if variant.startswith("semantic"):
        return "semantic_distortion"
    if variant.startswith("stealthy") or variant.startswith("injection"):
        return "stealthy_injection"
    return variant

def train_detectors(
    clean_embeddings: np.ndarray,
    *,
    seed: int = SEED,
    n_estimators: int = 200,
    n_neighbors: int = 20,
    contamination: str | float = "auto",
) -> Dict[str, object]:
    set_seed(seed)

    LOGGER.info(
        "Fitting IsolationForest n_estimators=%d contamination=%s on %d clean docs",
        n_estimators,
        contamination,
        clean_embeddings.shape[0],
    )
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    ).fit(clean_embeddings)

    LOGGER.info(
        "Fitting LOF n_neighbors=%d novelty=True contamination=%s",
        n_neighbors,
        contamination,
    )
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        novelty=True,
        contamination=contamination,
        n_jobs=-1,
    ).fit(clean_embeddings)

    return {ISOLATION_FOREST: iso, LOF: lof}

def _metrics(
    scores: np.ndarray,
    preds: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if labels.sum() == 0 or labels.sum() == len(labels):
        out["roc_auc"] = float("nan")
    else:
        out["roc_auc"] = float(roc_auc_score(labels, scores))
    out["precision"] = float(precision_score(labels, preds, zero_division=0))
    out["recall"] = float(recall_score(labels, preds, zero_division=0))
    out["f1"] = float(f1_score(labels, preds, zero_division=0))
    return out

def score_with(
    detector_name: str,
    detector: object,
    embeddings: np.ndarray,
    labels: np.ndarray,
    variant: str,
) -> DetectorResult:
    if detector_name == ISOLATION_FOREST:
        raw = detector.score_samples(embeddings)
    elif detector_name == LOF:
        raw = detector.score_samples(embeddings)
    else:
        raise ValueError(f"Unknown detector: {detector_name}")

    anomaly_scores = -raw
    preds = (detector.predict(embeddings) == -1).astype(int)

    metrics = _metrics(anomaly_scores, preds, labels)
    return DetectorResult(
        detector=detector_name,
        variant=variant,
        scores=anomaly_scores.astype(np.float32),
        preds=preds.astype(np.int8),
        labels=labels.astype(np.int8),
        **metrics,
    )

def save_scores(result: DetectorResult, scores_dir: Path = M2_SCORES) -> Path:
    scores_dir.mkdir(parents=True, exist_ok=True)
    out = scores_dir / f"{result.variant}_{result.detector}_scores.npy"
    np.save(out, result.scores)
    return out

def log_result(result: DetectorResult, notes: str = "") -> None:
    append_detection_row({
        "experiment_id": f"{result.variant}_{result.detector}",
        "attack_type": _attack_type_from_variant(result.variant),
        "poison_rate": _poison_rate_from_variant(result.variant),
        "detector": result.detector,
        "n_docs": int(result.labels.shape[0]),
        "n_poisoned_true": int(result.labels.sum()),
        "n_flagged": int(result.preds.sum()),
        "precision": round(result.precision, 4),
        "recall": round(result.recall, 4),
        "f1": round(result.f1, 4),
        "roc_auc": round(result.roc_auc, 4) if result.roc_auc == result.roc_auc else "",
        "threshold": result.threshold_desc,
        "notes": notes,
    })

def evaluate_variants(
    variants: List[str],
    *,
    seed: int = SEED,
    n_estimators: int = 200,
    n_neighbors: int = 20,
) -> List[DetectorResult]:
    clean_emb, _, _ = load_variant("clean")
    detectors = train_detectors(
        clean_emb,
        seed=seed,
        n_estimators=n_estimators,
        n_neighbors=n_neighbors,
    )

    results: List[DetectorResult] = []
    for variant in variants:
        emb, _, labels = load_variant(variant)
        if labels.sum() == 0:
            LOGGER.warning(
                "Variant '%s' has zero positive labels; detection metrics will be undefined.",
                variant,
            )
        for name, det in detectors.items():
            res = score_with(name, det, emb, labels, variant)
            LOGGER.info(
                "[%s @ %s] P=%.3f R=%.3f F1=%.3f AUC=%.3f flagged=%d/%d true_pos=%d",
                name,
                variant,
                res.precision,
                res.recall,
                res.f1,
                res.roc_auc,
                int(res.preds.sum()),
                len(res.preds),
                int(res.labels.sum()),
            )
            save_scores(res)
            log_result(res)
            results.append(res)
    return results
