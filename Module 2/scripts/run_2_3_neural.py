#!/usr/bin/env python3
"""Task 2.3 — Train and evaluate the supervised MLP classifier per variant."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_MODULE2_ROOT = Path(__file__).resolve().parents[1]
if str(_MODULE2_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE2_ROOT))

from src.detection.embeddings import discover_poisoned_variants  # noqa: E402
from src.detection.neural_classifier import evaluate_variants  # noqa: E402
from src.detection.utils import M2_POISONED_KB, setup_logging  # noqa: E402

LOGGER = logging.getLogger("run_2_3_neural")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 2.3 — Supervised MLP classifier.")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
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

    evaluate_variants(
        variants,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
