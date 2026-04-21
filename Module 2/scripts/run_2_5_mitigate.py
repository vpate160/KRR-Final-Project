from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_MODULE2_ROOT = Path(__file__).resolve().parents[1]
if str(_MODULE2_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE2_ROOT))

from src.detection.anomaly_detector import ISOLATION_FOREST, LOF
from src.detection.mitigation import run_mitigation
from src.detection.neural_classifier import NEURAL
from src.detection.utils import MODULE2_ROOT, setup_logging

LOGGER = logging.getLogger("run_2_5_mitigate")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 2.5 — Mitigation pipeline.")
    parser.add_argument("--variant", type=str, required=True, help="e.g. factual_10pct")
    parser.add_argument(
        "--detector",
        type=str,
        default=NEURAL,
        choices=[ISOLATION_FOREST, LOF, NEURAL],
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the detector's decision threshold. MLP only.",
    )
    parser.add_argument("--n-examples", type=int, default=1000)
    parser.add_argument("--ollama-model", type=str, default="llama3.1:8b")
    parser.add_argument("--ollama-base-url", type=str, default="http://localhost:11434")
    parser.add_argument(
        "--run-undefended",
        action="store_true",
        help="Also evaluate the poisoned-but-unfiltered KB for before/after.",
    )
    parser.add_argument(
        "--predictions-root",
        type=str,
        default=str(MODULE2_ROOT / "results" / "predictions"),
        help="Dir to write per-example predictions jsonl files. Pass empty to skip.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    predictions_root = Path(args.predictions_root) if args.predictions_root else None

    outcome = run_mitigation(
        variant=args.variant,
        detector=args.detector,
        n_examples=args.n_examples,
        threshold=args.threshold,
        run_undefended=args.run_undefended,
        ollama_model=args.ollama_model,
        ollama_base_url=args.ollama_base_url,
        predictions_root=predictions_root,
    )

    print("\n=== Mitigation Summary ===")
    print(f"variant: {outcome.variant}")
    print(f"detector: {outcome.detector}")
    print(f"flagged: {outcome.n_flagged} / kept: {outcome.n_kept}")
    print(f"filtered KB:          {outcome.filtered_kb_path}")
    print(f"filtered vectorstore: {outcome.filtered_vectorstore_path}")
    print(
        f"defended EM={outcome.defended_metrics.get('exact_match', 0.0):.4f} "
        f"F1={outcome.defended_metrics.get('token_f1', 0.0):.4f}"
    )
    if outcome.undefended_metrics is not None:
        print(
            f"undefended EM={outcome.undefended_metrics.get('exact_match', 0.0):.4f} "
            f"F1={outcome.undefended_metrics.get('token_f1', 0.0):.4f}"
        )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
