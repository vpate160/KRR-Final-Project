from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_MODULE2_ROOT = Path(__file__).resolve().parents[1]
if str(_MODULE2_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE2_ROOT))

from src.detection.embeddings import (
    discover_poisoned_variants,
    ensure_clean_labels,
    extract_and_save,
)
from src.detection.utils import (
    CLEAN_EMBEDDINGS,
    POISONED_KB_SEARCH_DIRS,
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
        help="Explicit variant names (e.g. factual_0.1). If omitted, auto-discover from "
        "data/poisoned_kb/<variant>.jsonl.",
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
                "No poisoned KBs found. Searched: %s. Module 1 ships variants as "
                "<variant>.jsonl (e.g. factual_0.1.jsonl) with a sibling "
                "<variant>_labels.npy.",
                [str(p) for p in POISONED_KB_SEARCH_DIRS],
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
