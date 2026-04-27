from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_MODULE2_ROOT = Path(__file__).resolve().parents[1]
if str(_MODULE2_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE2_ROOT))

from src.detection.embeddings import discover_poisoned_variants
from src.detection.llm_judge import evaluate_variants
from src.detection.utils import POISONED_KB_SEARCH_DIRS, setup_logging

LOGGER = logging.getLogger("run_2_4_llm_judge")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 2.4b — LLM-as-judge baseline.")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--ollama-model", type=str, default="llama3.1:8b")
    parser.add_argument("--ollama-base-url", type=str, default="http://localhost:11434")
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Stratified sample size per variant. Default: all docs.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
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
                "No poisoned variants found. Searched: %s. Pass --variants explicitly "
                "or copy Module 1's <variant>.jsonl files into one of those dirs.",
                [str(p) for p in POISONED_KB_SEARCH_DIRS],
            )
            return 2
        variants = [v for v, _ in discovered]

    evaluate_variants(
        variants,
        ollama_model=args.ollama_model,
        ollama_base_url=args.ollama_base_url,
        max_docs=args.max_docs,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
