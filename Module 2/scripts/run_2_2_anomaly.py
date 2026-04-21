#!/usr/bin/env python3
"""Task 2.2 — Run Isolation Forest + LOF on every poisoned variant."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_MODULE2_ROOT = Path(__file__).resolve().parents[1]
if str(_MODULE2_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE2_ROOT))

from src.detection.anomaly_detector import evaluate_variants  # noqa: E402
from src.detection.embeddings import discover_poisoned_variants  # noqa: E402
from src.detection.utils import M2_POISONED_KB, setup_logging  # noqa: E402

LOGGER = logging.getLogger("run_2_2_anomaly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 2.2 — Unsupervised anomaly detection.")
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="Variant names to score. Defaults to auto-discovered poisoned files.",
    )
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--n-neighbors", type=int, default=20)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    if args.variants:
        variants = list(args.variants)
    else:
        discovered = discover_poisoned_variants()
        if not discovered:
            LOGGER.error(
                "No poisoned variants found. Place files at %s or pass --variants.",
                M2_POISONED_KB,
            )
            return 2
        variants = [v for v, _ in discovered]

    LOGGER.info("Evaluating %d variants: %s", len(variants), variants)
    evaluate_variants(
        variants,
        n_estimators=args.n_estimators,
        n_neighbors=args.n_neighbors,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
