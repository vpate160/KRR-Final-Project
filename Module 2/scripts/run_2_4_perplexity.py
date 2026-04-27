from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_MODULE2_ROOT = Path(__file__).resolve().parents[1]
if str(_MODULE2_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE2_ROOT))

from src.detection.embeddings import discover_poisoned_variants
from src.detection.perplexity import evaluate_variants
from src.detection.utils import M2_POISONED_KB, setup_logging

LOGGER = logging.getLogger("run_2_4_perplexity")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 2.4a — Perplexity baseline.")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--model-name", type=str, default="gpt2")
    parser.add_argument(
        "--k-frac",
        type=float,
        default=None,
        help="Fraction of docs to flag. Defaults to the variant's known poison rate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=[None, "cpu", "cuda", "mps"],
    )
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
        model_name=args.model_name,
        k_frac=args.k_frac,
        device=args.device,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
