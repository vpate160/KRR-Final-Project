#!/usr/bin/env python3
"""Smoke test for Task 2.2 plumbing using synthetic embeddings.

Does NOT touch Module 3's real data files. Exercises:
- utils.append_detection_row (CSV writer)
- anomaly_detector.train_detectors + score_with (sklearn integration)
- Metric correctness on a dataset where clean and "poisoned" are separated.

Run after `pip install numpy scikit-learn` to validate before pulling the
full deps (torch / transformers / sentence-transformers).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

_MODULE2_ROOT = Path(__file__).resolve().parents[1]
if str(_MODULE2_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE2_ROOT))


def test_append_detection_row_to_tempfile() -> None:
    from src.detection import utils

    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "detection_metrics.csv"
        row = {
            "experiment_id": "smoke_test",
            "attack_type": "factual_swap",
            "poison_rate": 0.1,
            "detector": "isolation_forest",
            "n_docs": 400,
            "n_poisoned_true": 40,
            "n_flagged": 41,
            "precision": 0.9,
            "recall": 0.9,
            "f1": 0.9,
            "roc_auc": 0.95,
            "threshold": "auto",
            "notes": "smoke",
        }
        utils.append_detection_row(row, csv_path=csv_path)
        assert csv_path.exists()
        header, data = csv_path.read_text(encoding="utf-8").splitlines()
        assert header.startswith("timestamp_utc,"), f"bad header: {header}"
        assert "isolation_forest" in data
    print("OK test_append_detection_row_to_tempfile")


def test_train_detectors_and_score() -> None:
    """Synthetic clean + poisoned embeddings; both detectors should catch the shift."""
    from src.detection.anomaly_detector import (
        ISOLATION_FOREST,
        LOF,
        score_with,
        train_detectors,
    )

    rng = np.random.default_rng(0)
    n_clean = 400
    n_poison = 40
    clean_emb = rng.normal(0.0, 1.0, size=(n_clean, 768)).astype(np.float32)

    variant_emb = clean_emb.copy()
    shift = rng.normal(3.0, 0.5, size=(n_poison, 768)).astype(np.float32)
    variant_emb[-n_poison:] += shift
    labels = np.zeros(n_clean, dtype=np.int8)
    labels[-n_poison:] = 1

    detectors = train_detectors(clean_emb, n_estimators=80, n_neighbors=15)
    assert set(detectors.keys()) == {ISOLATION_FOREST, LOF}

    for name, det in detectors.items():
        result = score_with(name, det, variant_emb, labels, variant="factual_10pct")
        assert 0.0 <= result.precision <= 1.0
        assert 0.0 <= result.recall <= 1.0
        assert 0.0 <= result.f1 <= 1.0
        assert result.roc_auc > 0.8, f"{name} AUC too low: {result.roc_auc:.3f}"
        print(
            f"   [{name}] P={result.precision:.3f} R={result.recall:.3f} "
            f"F1={result.f1:.3f} AUC={result.roc_auc:.3f} "
            f"flagged={int(result.preds.sum())}/{len(result.preds)}"
        )
    print("OK test_train_detectors_and_score")


def test_variant_name_parsers() -> None:
    from src.detection.anomaly_detector import (
        _attack_type_from_variant,
        _poison_rate_from_variant,
    )

    cases = [
        ("factual_10pct", "factual_swap", 0.1),
        ("factual_30pct", "factual_swap", 0.3),
        ("semantic_10pct", "semantic_distortion", 0.1),
        ("semantic_30pct", "semantic_distortion", 0.3),
        ("stealthy_injection_5pct", "stealthy_injection", 0.05),
    ]
    for variant, expected_type, expected_rate in cases:
        got_type = _attack_type_from_variant(variant)
        got_rate = _poison_rate_from_variant(variant)
        assert got_type == expected_type, f"{variant}: attack_type got={got_type} want={expected_type}"
        assert abs(got_rate - expected_rate) < 1e-9, (
            f"{variant}: rate got={got_rate} want={expected_rate}"
        )
    print("OK test_variant_name_parsers")


def main() -> int:
    test_append_detection_row_to_tempfile()
    test_variant_name_parsers()
    test_train_detectors_and_score()
    print("\nALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
