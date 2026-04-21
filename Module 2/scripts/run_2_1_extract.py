#!/usr/bin/env python3
"""Task 2.1 — Extract embeddings for every KB variant.

Usage (from repo root):
    python "Module 2/scripts/run_2_1_extract.py"              # auto-discover poisoned files
    python "Module 2/scripts/run_2_1_extract.py" --variants factual_10pct semantic_30pct
    python "Module 2/scripts/run_2_1_extract.py" --force      # re-encode even if outputs exist

Clean embeddings are assumed to already exist at
    ../data/embeddings/clean_embeddings.npy  (produced by Vatsal's export_clean_embeddings.py)
This script only (re-)encodes poisoned KB variants staged under
    Module 2/data/poisoned_kb/poisoned_<variant>.jsonl
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make the src package importable when running as a standalone script.
_MODULE2_ROOT = Path(__file__).resolve().parents[1]
if str(_MODULE2_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE2_ROOT))

from src.detection.embeddings import (  # noqa: E402
    discover_poisoned_variants,
    ensure_clean_labels,
    extract_and_save,
)
from src.detection.utils import (  # noqa: E402
    CLEAN_EMBEDDINGS,
    M2_POISONED_KB,
    setup_logging,
    variant_paths,
)

LOGGER = logging.getLogger("run_2_1_extract")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 2.1 — Encode poisoned KB variants.")
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="Explicit variant names (e.g. factual_10pct). If omitted, auto-discover from "
        "Module 2/data/poisoned_kb/poisoned_*.jsonl.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-encode even if outputs already exist.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    if not CLEAN_EMBEDDINGS.exists():
        LOGGER.error(
            "Clean embeddings not found at %s. Pull latest main from Vatsal's repo first.",
            CLEAN_EMBEDDINGS,
        )
        return 2
    LOGGER.info("Clean embeddings already present at %s (reused).", CLEAN_EMBEDDINGS)
    ensure_clean_labels()

    if args.variants:
        targets = [(v, variant_paths(v)["kb"]) for v in args.variants]
    else:
        targets = discover_poisoned_variants()
        if not targets:
            LOGGER.warning(
                "No poisoned KBs found in %s. Copy Hardik's files there as "
                "poisoned_<variant>.jsonl (e.g. poisoned_factual_10pct.jsonl).",
                M2_POISONED_KB,
            )
            return 0

    failures: list[str] = []
    for variant, kb_path in targets:
        try:
            LOGGER.info("=== Encoding variant '%s' from %s ===", variant, kb_path)
            extract_and_save(
                kb_path=kb_path,
                variant=variant,
                batch_size=args.batch_size,
                device=args.device,
                force=args.force,
            )
        except Exception as exc:
            LOGGER.exception("Failed to encode variant '%s': %s", variant, exc)
            failures.append(variant)

    if failures:
        LOGGER.error("Failures: %s", failures)
        return 1
    LOGGER.info("All variants encoded successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
