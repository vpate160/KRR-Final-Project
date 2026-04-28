from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import auc as sklearn_auc
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE2_ROOT = REPO_ROOT / "Module 2"
ROOT_METRICS = REPO_ROOT / "results" / "metrics.csv"
DETECTION_METRICS = MODULE2_ROOT / "results" / "detection_metrics.csv"
MITIGATION_SUMMARY = MODULE2_ROOT / "results" / "mitigation_filter_summary.csv"
EMBEDDINGS_DIR = MODULE2_ROOT / "data" / "embeddings"
SCORES_DIR = MODULE2_ROOT / "results" / "scores"
FIGURES_DIR = REPO_ROOT / "results" / "figures"
TABLES_DIR = REPO_ROOT / "results" / "tables"
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "results" / "dashboard.ipynb"
FAILURE_DIR = REPO_ROOT / "results" / "failure_analysis"
MODULE2_VIS_DIR = MODULE2_ROOT / "results" / "visualizations"
COVERAGE_REPORT = REPO_ROOT / "results" / "module4_task_coverage.md"
REPORT_PATH = REPO_ROOT / "results" / "module4_visualization_report.html"
HANDOFF_PATH = REPO_ROOT / "results" / "module4_handoff_action_items.md"
LANCE_HANDOFF_PATH = REPO_ROOT / "results" / "module4_lance_handoff_package.md"
CHANGE_LOG_DIR = REPO_ROOT / "logs"

SEED = 42
VARIANTS = ["factual_0.1", "factual_0.2", "factual_0.3", "semantic_0.1", "semantic_0.3"]
DETECTORS = ["isolation_forest", "lof", "neural_classifier", "perplexity", "llm_judge"]
DETECTOR_LABELS = {
    "isolation_forest": "Isolation Forest",
    "lof": "LOF",
    "neural_classifier": "Neural Classifier",
    "perplexity": "GPT-2 Perplexity",
    "llm_judge": "LLM Judge",
}
COLORS = {
    "exact_match": "#277da1",
    "token_f1": "#2a9d8f",
    "precision": "#277da1",
    "recall": "#d55e00",
    "f1": "#2a9d8f",
    "roc_auc": "#6d597a",
    "isolation_forest": "#277da1",
    "lof": "#577590",
    "neural_classifier": "#2a9d8f",
    "perplexity": "#d55e00",
    "llm_judge": "#6d597a",
    "factual_swap": "#277da1",
    "semantic_distortion": "#d55e00",
    "clean baseline F1": "#4f5d75",
    "clean baseline EM": "#8d99ae",
    "factual swap undefended F1": "#277da1",
    "factual swap mitigated F1": "#8ecae6",
    "semantic distortion undefended F1": "#d55e00",
    "semantic distortion mitigated F1": "#f4a261",
    "undefended": "#4f5d75",
    "threshold_0.5": "#d55e00",
    "topK_capped": "#2a9d8f",
}


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    FAILURE_DIR.mkdir(parents=True, exist_ok=True)


def read_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rag = pd.read_csv(ROOT_METRICS) if ROOT_METRICS.exists() else pd.DataFrame()
    detection = pd.read_csv(DETECTION_METRICS) if DETECTION_METRICS.exists() else pd.DataFrame()
    mitigation = pd.read_csv(MITIGATION_SUMMARY) if MITIGATION_SUMMARY.exists() else pd.DataFrame()
    if not detection.empty:
        detection["variant"] = detection["experiment_id"].apply(variant_from_experiment)
        detection["poison_rate"] = pd.to_numeric(detection["poison_rate"], errors="coerce")
        for col in ["precision", "recall", "f1", "roc_auc"]:
            detection[col] = pd.to_numeric(detection[col], errors="coerce")
    if not rag.empty:
        for col in ["poison_rate", "exact_match", "token_f1"]:
            rag[col] = pd.to_numeric(rag[col], errors="coerce")
    if not mitigation.empty:
        for col in ["filter_recall", "filter_precision"]:
            mitigation[col] = pd.to_numeric(mitigation[col], errors="coerce")
        mitigation["attack_type"] = mitigation["variant"].apply(attack_type_from_variant)
        mitigation["poison_rate"] = mitigation["variant"].apply(poison_rate_from_variant)
    return rag, detection, mitigation


def latest_rag_rows(rag: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest/highest-n row for each RAG condition."""
    if rag.empty or "n_examples" not in rag.columns:
        return rag.copy()
    latest = rag.copy()
    latest["n_examples"] = pd.to_numeric(latest["n_examples"], errors="coerce")
    if "timestamp_utc" in latest.columns:
        latest["_timestamp_sort"] = pd.to_datetime(latest["timestamp_utc"], errors="coerce")
    else:
        latest["_timestamp_sort"] = pd.NaT
    group_cols = [c for c in ["attack_type", "poison_rate", "detector", "dataset"] if c in latest.columns]
    if not group_cols:
        return latest.drop(columns=["_timestamp_sort"], errors="ignore")
    latest = (
        latest.sort_values(["n_examples", "_timestamp_sort"], na_position="first")
        .groupby(group_cols, dropna=False, as_index=False)
        .tail(1)
        .drop(columns=["_timestamp_sort"], errors="ignore")
    )
    sort_cols = [c for c in ["attack_type", "poison_rate", "detector", "n_examples"] if c in latest.columns]
    return latest.sort_values(sort_cols).reset_index(drop=True)


def variant_from_experiment(experiment_id: str) -> str:
    for variant in VARIANTS:
        if str(experiment_id).startswith(f"{variant}_"):
            return variant
    return ""


def attack_type_from_variant(variant: str) -> str:
    if variant.startswith("factual"):
        return "factual_swap"
    if variant.startswith("semantic"):
        return "semantic_distortion"
    return "unknown"


def poison_rate_from_variant(variant: str) -> float:
    try:
        return float(variant.split("_")[-1])
    except ValueError:
        return math.nan


def variant_from_rag_row(row: pd.Series) -> str:
    attack_type = str(row.get("attack_type", ""))
    rate = row.get("poison_rate", math.nan)
    try:
        rate_text = f"{float(rate):.1f}"
    except (TypeError, ValueError):
        rate_text = str(rate)
    if attack_type == "factual_swap":
        return f"factual_{rate_text}"
    if attack_type == "semantic_distortion":
        return f"semantic_{rate_text}"
    return str(row.get("variant", ""))


def rag_strategy_rows(rag: pd.DataFrame) -> List[Dict[str, Any]]:
    latest = latest_rag_rows(rag)
    if latest.empty:
        return []
    recovery = latest[latest.get("attack_type", "") != "clean"].copy()
    if recovery.empty:
        return []
    recovery["variant"] = recovery.apply(variant_from_rag_row, axis=1)
    fallback_undefended = recovery[recovery["detector"] == "none"]["token_f1"]
    baseline = float(fallback_undefended.iloc[0]) if not fallback_undefended.empty else math.nan

    rows: List[Dict[str, Any]] = []
    for variant in VARIANTS:
        variant_rows = recovery[recovery["variant"] == variant]
        undefended = variant_rows[variant_rows["detector"] == "none"]["token_f1"]
        threshold = variant_rows[variant_rows["detector"] == "mitigated"]["token_f1"]
        topk = variant_rows[variant_rows["detector"] == "mitigated_topK"]["token_f1"]
        threshold_missing = threshold.empty
        rows.append(
            {
                "variant": variant,
                "poison_rate": poison_rate_from_variant(variant),
                "undefended": float(undefended.iloc[0]) if not undefended.empty else baseline,
                "threshold_0.5": float(threshold.iloc[0]) if not threshold.empty else 0.0,
                "topK_capped": float(topk.iloc[0]) if not topk.empty else math.nan,
                "threshold_note": "empty KB / no row" if threshold_missing and variant == "factual_0.2" else "",
                "undefended_note": "reported aggregate baseline" if undefended.empty and math.isfinite(baseline) else "",
            }
        )
    return rows


def mitigation_strategy_summary(rag: pd.DataFrame, mitigation: pd.DataFrame) -> pd.DataFrame:
    latest = latest_rag_rows(rag)
    recovery = latest[latest.get("attack_type", "") != "clean"].copy() if not latest.empty else pd.DataFrame()
    if not recovery.empty:
        recovery["variant"] = recovery.apply(variant_from_rag_row, axis=1)

    rows: List[Dict[str, Any]] = []
    for variant in VARIANTS:
        rate = poison_rate_from_variant(variant)
        threshold_rag = recovery[(recovery.get("variant", "") == variant) & (recovery.get("detector", "") == "mitigated")] if not recovery.empty else pd.DataFrame()
        topk_rag = recovery[(recovery.get("variant", "") == variant) & (recovery.get("detector", "") == "mitigated_topK")] if not recovery.empty else pd.DataFrame()
        mrow = mitigation[mitigation["variant"] == variant].head(1) if not mitigation.empty else pd.DataFrame()
        rows.append(
            {
                "variant": variant,
                "strategy": "threshold_0.5",
                "poison_rate": rate,
                "filter_recall": float(mrow.iloc[0]["filter_recall"]) if not mrow.empty else math.nan,
                "filter_precision": float(mrow.iloc[0]["filter_precision"]) if not mrow.empty else math.nan,
                "rag_token_f1": float(threshold_rag.iloc[0]["token_f1"]) if not threshold_rag.empty else math.nan,
                "notes": "naive threshold; empty KB for factual_0.2" if variant == "factual_0.2" else "naive threshold",
            }
        )
        if not topk_rag.empty:
            rows.append(
                {
                    "variant": variant,
                    "strategy": "topK_capped",
                    "poison_rate": rate,
                    "filter_recall": math.nan,
                    "filter_precision": math.nan,
                    "rag_token_f1": float(topk_rag.iloc[0]["token_f1"]),
                    "notes": "capped at expected poison rate; preserves KB integrity",
                }
            )
    return pd.DataFrame(rows)


def write_tables(rag: pd.DataFrame, detection: pd.DataFrame, mitigation: pd.DataFrame) -> Dict[str, Path]:
    outputs: Dict[str, Path] = {}
    rag = latest_rag_rows(rag)

    rag_cols = [
        "timestamp_utc",
        "attack_type",
        "poison_rate",
        "detector",
        "dataset",
        "n_examples",
        "exact_match",
        "token_f1",
        "generation_backend",
        "ollama_model",
    ]
    rag_table = rag[[c for c in rag_cols if c in rag.columns]].copy() if not rag.empty else pd.DataFrame()
    outputs["rag_accuracy"] = TABLES_DIR / "module4_table1_rag_accuracy.csv"
    rag_table.to_csv(outputs["rag_accuracy"], index=False)

    det_cols = [
        "variant",
        "attack_type",
        "poison_rate",
        "detector",
        "n_docs",
        "n_poisoned_true",
        "n_flagged",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "threshold",
        "notes",
    ]
    det_table = detection[[c for c in det_cols if c in detection.columns]].copy() if not detection.empty else pd.DataFrame()
    outputs["detector_performance"] = TABLES_DIR / "module4_table2_detector_performance.csv"
    det_table.to_csv(outputs["detector_performance"], index=False)

    outputs["mitigation_summary"] = TABLES_DIR / "module4_table3_mitigation_filter_summary.csv"
    mitigation_strategy_summary(rag, mitigation).to_csv(outputs["mitigation_summary"], index=False)
    return outputs


def svg_wrap(width: int, height: int, body: str, title: str, subtitle: str = "") -> str:
    safe_title = html.escape(title)
    safe_subtitle = html.escape(subtitle)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="28" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#1b1f24">{safe_title}</text>
  <text x="28" y="58" font-family="Arial, sans-serif" font-size="13" fill="#5d6673">{safe_subtitle}</text>
  {body}
</svg>
"""


def scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if not math.isfinite(value):
        return dst_min
    if src_max == src_min:
        return (dst_min + dst_max) / 2.0
    return dst_min + ((value - src_min) / (src_max - src_min)) * (dst_max - dst_min)


def axis_lines(x0: int, y0: int, x1: int, y1: int) -> str:
    return (
        f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#d9ded8" stroke-width="1"/>'
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#d9ded8" stroke-width="1"/>'
    )


def legend(items: Sequence[Tuple[str, str]], x: int, y: int) -> str:
    parts: List[str] = []
    for i, (label, color) in enumerate(items):
        yy = y + i * 22
        parts.append(f'<circle cx="{x}" cy="{yy}" r="5" fill="{color}"/>')
        parts.append(
            f'<text x="{x + 14}" y="{yy + 4}" font-family="Arial, sans-serif" font-size="12" fill="#5d6673">{html.escape(label)}</text>'
        )
    return "\n".join(parts)


def write_grouped_bar_svg(
    rows: List[Dict[str, Any]],
    *,
    out_path: Path,
    title: str,
    subtitle: str,
    group_key: str,
    value_keys: Sequence[str],
    value_labels: Optional[Dict[str, str]] = None,
    value_max: float = 1.0,
    width: int = 1180,
    height: int = 620,
) -> Path:
    x0, y0, x1, y1 = 90, 90, width - 40, height - 92
    groups = [str(row[group_key]) for row in rows]
    group_w = (x1 - x0) / max(len(rows), 1)
    bar_gap = 4
    bar_w = min(28.0, (group_w - 18) / max(len(value_keys), 1) - bar_gap)
    body: List[str] = [axis_lines(x0, y0, x1, y1)]

    for tick in np.linspace(0, value_max, 6):
        ty = scale(float(tick), 0.0, value_max, y1, y0)
        tick_label = f"{tick:.2f}" if value_max <= 0.3 else f"{tick:.1f}"
        body.append(f'<line x1="{x0 - 4}" y1="{ty:.1f}" x2="{x1}" y2="{ty:.1f}" stroke="#eef0ed" stroke-width="1"/>')
        body.append(
            f'<text x="{x0 - 12}" y="{ty + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#5d6673">{tick_label}</text>'
        )

    for i, row in enumerate(rows):
        base_x = x0 + i * group_w + group_w / 2.0
        total_w = len(value_keys) * bar_w + (len(value_keys) - 1) * bar_gap
        for j, key in enumerate(value_keys):
            val = float(row.get(key, 0.0) or 0.0)
            color = COLORS.get(key, "#277da1")
            bx = base_x - total_w / 2.0 + j * (bar_w + bar_gap)
            by = scale(val, 0.0, value_max, y1, y0)
            body.append(
                f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{max(0, y1 - by):.1f}" fill="{color}" fill-opacity="0.86"/>'
            )
        body.append(
            f'<text x="{base_x:.1f}" y="{y1 + 28}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#5d6673">{html.escape(groups[i])}</text>'
        )

    label_map = value_labels or {key: key for key in value_keys}
    body.append(legend([(label_map[key], COLORS.get(key, "#277da1")) for key in value_keys], x0, height - 44))
    out_path.write_text(svg_wrap(width, height, "\n".join(body), title, subtitle), encoding="utf-8")
    return out_path


def write_line_svg(
    series: Dict[str, List[Tuple[float, float]]],
    *,
    out_path: Path,
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
    y_max: float = 1.0,
    width: int = 980,
    height: int = 600,
) -> Path:
    x0, y0, x1, y1 = 86, 88, width - 210, height - 82
    all_x = [x for points in series.values() for x, _ in points]
    min_x = min(all_x) if all_x else 0.0
    max_x = max(all_x) if all_x else 1.0
    if min_x == max_x:
        min_x, max_x = 0.0, max(1.0, max_x)

    body: List[str] = [axis_lines(x0, y0, x1, y1)]
    for tick in np.linspace(0, y_max, 6):
        ty = scale(float(tick), 0.0, y_max, y1, y0)
        tick_label = f"{tick:.2f}" if y_max <= 0.3 else f"{tick:.1f}"
        body.append(f'<line x1="{x0 - 4}" y1="{ty:.1f}" x2="{x1}" y2="{ty:.1f}" stroke="#eef0ed" stroke-width="1"/>')
        body.append(
            f'<text x="{x0 - 12}" y="{ty + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#5d6673">{tick_label}</text>'
        )
    for tick in sorted(set(round(x, 2) for x in all_x)):
        tx = scale(tick, min_x, max_x, x0, x1)
        body.append(
            f'<text x="{tx:.1f}" y="{y1 + 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#5d6673">{tick:g}</text>'
        )

    legend_items: List[Tuple[str, str]] = []
    for name, points in series.items():
        color = COLORS.get(name, "#277da1")
        sorted_points = sorted(points)
        poly = " ".join(
            f"{scale(x, min_x, max_x, x0, x1):.1f},{scale(y, 0.0, y_max, y1, y0):.1f}"
            for x, y in sorted_points
        )
        if len(sorted_points) == 1:
            x, y = sorted_points[0]
            body.append(
                f'<circle cx="{scale(x, min_x, max_x, x0, x1):.1f}" cy="{scale(y, 0.0, y_max, y1, y0):.1f}" r="5" fill="{color}"/>'
            )
        else:
            body.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="3"/>')
            for x, y in sorted_points:
                body.append(
                    f'<circle cx="{scale(x, min_x, max_x, x0, x1):.1f}" cy="{scale(y, 0.0, y_max, y1, y0):.1f}" r="4" fill="{color}"/>'
                )
        legend_items.append((name.replace("_", " "), color))

    body.append(
        f'<text x="{(x0 + x1) / 2:.1f}" y="{height - 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#5d6673">{html.escape(x_label)}</text>'
    )
    body.append(
        f'<text x="22" y="{(y0 + y1) / 2:.1f}" transform="rotate(-90 22 {(y0 + y1) / 2:.1f})" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#5d6673">{html.escape(y_label)}</text>'
    )
    body.append(legend(legend_items, x1 + 26, y0 + 6))
    out_path.write_text(svg_wrap(width, height, "\n".join(body), title, subtitle), encoding="utf-8")
    return out_path


def write_detector_figures(detection: pd.DataFrame) -> List[Path]:
    outputs: List[Path] = []
    if detection.empty:
        return outputs

    rows = []
    for _, row in detection.sort_values(["variant", "detector"]).iterrows():
        rows.append(
            {
                "label": f"{row['variant']}\n{row['detector']}",
                "variant_detector": f"{row['variant']} {short_detector(row['detector'])}",
                "f1": row.get("f1", 0.0),
                "roc_auc": 0.0 if pd.isna(row.get("roc_auc")) else row.get("roc_auc", 0.0),
            }
        )
    outputs.append(
        write_grouped_bar_svg(
            rows,
            out_path=FIGURES_DIR / "detector_f1_auc_by_variant.svg",
            title="Detector F1 and AUC by Variant",
            subtitle="AUCs cluster near 0.5; LLM-judge AUC is unavailable because it produced binary sampled verdicts.",
            group_key="variant_detector",
            value_keys=["f1", "roc_auc"],
            value_labels={"f1": "F1", "roc_auc": "AUC"},
            width=1320,
            height=660,
        )
    )

    subset = detection[detection["poison_rate"].isin([0.1, 0.3])].copy()
    if not subset.empty:
        rows = []
        for _, row in subset.sort_values(["attack_type", "poison_rate", "detector"]).iterrows():
            rows.append(
                {
                    "label": f"{row['variant']} {row['detector']}",
                    "variant_detector": f"{row['variant']} {short_detector(row['detector'])}",
                    "precision": row.get("precision", 0.0),
                    "recall": row.get("recall", 0.0),
                    "f1": row.get("f1", 0.0),
                }
            )
        outputs.append(
            write_grouped_bar_svg(
                rows,
                out_path=FIGURES_DIR / "detector_precision_recall_f1_10_30.svg",
                title="Precision, Recall, and F1 at 10% and 30% Poisoning",
                subtitle="Includes available detector rows at the report-requested low/high contamination rates.",
                group_key="variant_detector",
                value_keys=["precision", "recall", "f1"],
                value_labels={"precision": "Precision", "recall": "Recall", "f1": "F1"},
                width=1320,
                height=660,
            )
        )
    return outputs


def short_detector(detector: str) -> str:
    mapping = {
        "isolation_forest": "IF",
        "lof": "LOF",
        "neural_classifier": "MLP",
        "perplexity": "PPL",
        "llm_judge": "LLM",
    }
    return mapping.get(detector, detector)


def write_rag_accuracy_figure(rag: pd.DataFrame) -> Path:
    out_path = FIGURES_DIR / "rag_accuracy_vs_contamination_available.svg"
    rows = rag_strategy_rows(rag)
    if not rows:
        body = '<text x="80" y="130" font-family="Arial, sans-serif" font-size="16" fill="#5d6673">No RAG metrics found in results/metrics.csv.</text>'
        out_path.write_text(svg_wrap(900, 260, body, "RAG Strategy Comparison", "Blocked: results/metrics.csv has no usable rows."), encoding="utf-8")
        return out_path
    return write_grouped_bar_svg(
        rows,
        out_path=out_path,
        title="RAG Strategy Comparison",
        subtitle="Naive threshold mitigation can collapse the KB; capped top-K preserves stability but remains neutral versus undefended.",
        group_key="variant",
        value_keys=["undefended", "threshold_0.5", "topK_capped"],
        value_labels={"undefended": "Undefended", "threshold_0.5": "Threshold=0.5", "topK_capped": "Top-K capped"},
        value_max=0.12,
        width=1180,
        height=620,
    )


def write_mitigation_figure(mitigation: pd.DataFrame) -> Optional[Path]:
    if mitigation.empty:
        return None
    series: Dict[str, List[Tuple[float, float]]] = {}
    for attack_type, group in mitigation.groupby("attack_type"):
        series[f"{attack_type} recall"] = [(float(r["poison_rate"]), float(r["filter_recall"])) for _, r in group.iterrows()]
        series[f"{attack_type} precision"] = [(float(r["poison_rate"]), float(r["filter_precision"])) for _, r in group.iterrows()]
    local_colors = {
        "factual_swap recall": "#277da1",
        "factual_swap precision": "#8ecae6",
        "semantic_distortion recall": "#d55e00",
        "semantic_distortion precision": "#f4a261",
    }
    COLORS.update(local_colors)
    return write_line_svg(
        series,
        out_path=FIGURES_DIR / "mitigation_filter_precision_recall.svg",
        title="Mitigation Filter Precision and Recall",
        subtitle="Threshold filtering has high recall but low precision, motivating the safer capped top-K comparison.",
        x_label="poison rate",
        y_label="filter score",
    )


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ["arialbd.ttf" if bold else "arial.ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _hex_to_rgb(color: str) -> Tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def _blend(color: str, alpha: float, bg: Tuple[int, int, int] = (255, 255, 255)) -> Tuple[int, int, int]:
    rgb = _hex_to_rgb(color)
    return tuple(int(alpha * rgb[i] + (1.0 - alpha) * bg[i]) for i in range(3))


def _png_scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return (dst_min + dst_max) / 2.0
    return dst_min + (float(value) - src_min) / (src_max - src_min) * (dst_max - dst_min)


def _draw_centered(draw: ImageDraw.ImageDraw, xy: Tuple[float, float], text: str, font: ImageFont.ImageFont, fill: str) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - (bbox[2] - bbox[0]) / 2, xy[1] - (bbox[3] - bbox[1]) / 2), text, font=font, fill=fill)


def write_embedding_small_multiples_png() -> Optional[Path]:
    projection_path = MODULE2_VIS_DIR / "embedding_projection.csv"
    if not projection_path.exists():
        return None

    projection = pd.read_csv(projection_path)
    metadata_path = MODULE2_VIS_DIR / "embedding_projection_metadata.json"
    method = "embedding"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        method = str(metadata.get("method") or metadata.get("requested_method") or "embedding")

    width, height = 2400, 1600
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    colors = {
        "clean_reference": "#8a9099",
        "clean_variant": "#277da1",
        "poisoned": "#d55e00",
    }
    title_font = _font(38, bold=True)
    subtitle_font = _font(22)
    label_font = _font(24, bold=True)
    note_font = _font(22)
    draw.text((70, 42), f"Embedding-Space Small Multiples ({method.upper()} projection)", font=title_font, fill="#1b1f24")
    draw.text((70, 92), "All five poisoning variants show poisoned documents overlapping clean documents.", font=subtitle_font, fill="#5d6673")
    panels = [
        (80, 160, 760, 570),
        (860, 160, 1540, 570),
        (1640, 160, 2320, 570),
        (80, 700, 760, 1110),
        (860, 700, 1540, 1110),
    ]
    ref = projection[projection["row_type"] == "clean_reference"]
    for i, variant in enumerate(VARIANTS):
        x0, y0, x1, y1 = panels[i]
        draw.rounded_rectangle((x0, y0, x1, y1), radius=12, outline="#d9ded8", width=2, fill="#fbfbf8")
        draw.text((x0 + 18, y0 + 14), variant, font=label_font, fill="#1b1f24")
        var = projection[(projection["row_type"] == "variant_doc") & (projection["variant"] == variant)]
        clean_docs = var[var["is_poisoned"].astype(int) == 0]
        poison_docs = var[var["is_poisoned"].astype(int) == 1]
        plot_rows = [(ref, colors["clean_reference"], 0.18, 2), (clean_docs, colors["clean_variant"], 0.45, 3), (poison_docs, colors["poisoned"], 0.88, 5)]
        xs = pd.concat([ref["x"], var["x"]])
        ys = pd.concat([ref["y"], var["y"]])
        xmin, xmax = float(xs.min()), float(xs.max())
        ymin, ymax = float(ys.min()), float(ys.max())
        pad_x = max((xmax - xmin) * 0.08, 1.0)
        pad_y = max((ymax - ymin) * 0.08, 1.0)
        for rows_df, color, alpha, radius in plot_rows:
            fill = _blend(color, alpha)
            for _, row in rows_df.iterrows():
                px = _png_scale(row["x"], xmin - pad_x, xmax + pad_x, x0 + 34, x1 - 34)
                py = _png_scale(row["y"], ymin - pad_y, ymax + pad_y, y1 - 34, y0 + 58)
                draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=fill)

    legend_x, legend_y = 1640, 725
    draw.rounded_rectangle((legend_x, legend_y, 2320, 1110), radius=12, outline="#d9ded8", width=2, fill="#ffffff")
    legend_items = [("clean reference", colors["clean_reference"], 0.35), ("clean in poisoned KB", colors["clean_variant"], 0.75), ("poisoned", colors["poisoned"], 0.9)]
    for i, (label, color, alpha) in enumerate(legend_items):
        yy = legend_y + 60 + i * 58
        draw.ellipse((legend_x + 42, yy - 11, legend_x + 64, yy + 11), fill=_blend(color, alpha))
        draw.text((legend_x + 84, yy - 16), label, font=note_font, fill="#1b1f24")
    draw.text((legend_x + 42, legend_y + 250), "Overlap is the finding:", font=_font(24, bold=True), fill="#1b1f24")
    draw.multiline_text((legend_x + 42, legend_y + 288), "poisoned points remain\ninside clean neighborhoods.", font=note_font, fill="#5d6673", spacing=8)
    out_path = FIGURES_DIR / "umap_small_multiples.png"
    image.save(out_path, dpi=(200, 200))
    return out_path


def write_rag_accuracy_png(rag: pd.DataFrame) -> Optional[Path]:
    rows = rag_strategy_rows(rag)
    if not rows:
        return None

    strategies = ["undefended", "threshold_0.5", "topK_capped"]
    labels = {
        "undefended": "Undefended",
        "threshold_0.5": "Threshold=0.5",
        "topK_capped": "Top-K capped",
    }
    values = [
        float(row[key])
        for row in rows
        for key in strategies
        if key in row and math.isfinite(float(row[key]))
    ]
    ymax = max(0.09, max(values) * 1.25 if values else 0.09)

    width, height = 2200, 1250
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(36, bold=True)
    label_font = _font(22)
    small_font = _font(18)
    note_font = _font(20)
    draw.text((70, 42), "RAG Strategy Comparison: Naive vs. Safe Mitigation", font=title_font, fill="#1b1f24")
    draw.text((70, 92), "Top-K capped mitigation prevents KB collapse but remains neutral versus undefended retrieval.", font=label_font, fill="#5d6673")

    x0, y0, x1, y1 = 170, 185, 1680, 920
    for tick in np.linspace(0, ymax, 6):
        ty = _png_scale(tick, 0, ymax, y1, y0)
        draw.line((x0, ty, x1, ty), fill="#eef0ed", width=2)
        draw.text((70, ty - 12), f"{tick:.2f}", font=small_font, fill="#5d6673")
    draw.line((x0, y1, x1, y1), fill="#d9ded8", width=3)
    draw.line((x0, y0, x0, y1), fill="#d9ded8", width=3)

    group_step = (x1 - x0) / len(rows)
    bar_w = min(78, group_step / 5.0)
    offsets = [-bar_w * 1.2, 0, bar_w * 1.2]
    for idx, row in enumerate(rows):
        cx = x0 + group_step * idx + group_step / 2
        for j, key in enumerate(strategies):
            value = float(row.get(key, math.nan))
            if not math.isfinite(value):
                continue
            top = _png_scale(value, 0, ymax, y1, y0)
            left = cx + offsets[j] - bar_w / 2
            right = cx + offsets[j] + bar_w / 2
            color = COLORS[key]
            if value <= 0:
                draw.rectangle((left, y1 - 4, right, y1), fill=_blend(color, 0.85))
                draw.text((left - 8, y1 - 34), "0.000", font=small_font, fill="#1b1f24")
            else:
                draw.rectangle((left, top, right, y1), fill=_blend(color, 0.85))
                _draw_centered(draw, ((left + right) / 2, top - 24), f"{value:.3f}", small_font, "#1b1f24")

        label = str(row["variant"]).replace("_", "\n")
        _draw_centered(draw, (cx, y1 + 42), label, small_font, "#1b1f24")
        if row.get("threshold_note"):
            draw.multiline_text((cx - 72, y1 + 92), "threshold\nempty KB", font=_font(15), fill="#9a3412", spacing=3)

    draw.text(((x0 + x1) // 2 - 80, y1 + 132), "Poisoning variant", font=label_font, fill="#5d6673")
    draw.text((24, (y0 + y1) // 2), "Token-F1", font=label_font, fill="#5d6673")

    legend_x, legend_y = 1755, 195
    draw.rounded_rectangle((legend_x - 26, legend_y - 34, 2130, legend_y + 275), radius=12, outline="#d9ded8", width=2, fill="#ffffff")
    for i, key in enumerate(strategies):
        yy = legend_y + i * 54
        draw.rectangle((legend_x, yy - 14, legend_x + 36, yy + 14), fill=_blend(COLORS[key], 0.85))
        draw.text((legend_x + 54, yy - 15), labels[key], font=small_font, fill="#1b1f24")
    draw.text((legend_x, legend_y + 180), "Main takeaway:", font=_font(22, bold=True), fill="#1b1f24")
    draw.multiline_text(
        (legend_x, legend_y + 215),
        "The capped defense fixes\nKB collapse, but it does\nnot improve answer F1.",
        font=note_font,
        fill="#5d6673",
        spacing=6,
    )

    draw.multiline_text(
        (170, 1100),
        "Naive thresholding is the destructive comparison case: it can remove too much clean evidence and fully collapses factual_0.2.\n"
        "Top-K capped mitigation is the safer engineering fix: all five variants run without KB collapse.\n"
        "Because the embeddings overlap, token-F1 remains essentially identical to undefended retrieval.",
        font=note_font,
        fill="#5d6673",
        spacing=8,
    )
    out_path = FIGURES_DIR / "rag_accuracy_curves.png"
    image.save(out_path, dpi=(200, 200))
    return out_path


def write_detector_comparison_png(detection: pd.DataFrame) -> Optional[Path]:
    if detection.empty or "roc_auc" not in detection:
        return None
    auc_rows = detection.dropna(subset=["roc_auc"]).copy()
    if auc_rows.empty:
        return None

    detector_order = ["isolation_forest", "lof", "neural_classifier", "perplexity"]
    means = auc_rows.groupby("detector")["roc_auc"].mean().reindex(detector_order).dropna()
    width, height = 1800, 1100
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(36, bold=True)
    label_font = _font(22)
    small_font = _font(18)
    draw.text((70, 42), "Detector Comparison: AUCs Hover Near Random", font=title_font, fill="#1b1f24")
    draw.text((70, 92), "Bars show mean AUC by detector; dots are variant-level AUCs.", font=label_font, fill="#5d6673")
    x0, y0, x1, y1 = 150, 165, 1420, 840
    ymin, ymax = 0.42, 0.63
    for tick in np.linspace(ymin, ymax, 6):
        ty = _png_scale(tick, ymin, ymax, y1, y0)
        draw.line((x0, ty, x1, ty), fill="#eef0ed", width=2)
        draw.text((70, ty - 12), f"{tick:.2f}", font=small_font, fill="#5d6673")
    draw.line((x0, y1, x1, y1), fill="#d9ded8", width=3)
    draw.line((x0, y0, x0, y1), fill="#d9ded8", width=3)
    random_y = _png_scale(0.5, ymin, ymax, y1, y0)
    for xx in range(x0, x1, 28):
        draw.line((xx, random_y, xx + 14, random_y), fill="#9a3412", width=3)
    draw.text((x1 - 230, random_y - 30), "random guess AUC=0.5", font=small_font, fill="#9a3412")
    bar_w = 170
    step = (x1 - x0) / len(means)
    for idx, detector in enumerate(means.index):
        cx = x0 + step * idx + step / 2
        top = _png_scale(float(means.loc[detector]), ymin, ymax, y1, y0)
        color = COLORS.get(detector, "#277da1")
        draw.rectangle((cx - bar_w / 2, top, cx + bar_w / 2, y1), fill=_blend(color, 0.82))
        _draw_centered(draw, (cx, top - 22), f"{means.loc[detector]:.3f}", small_font, "#1b1f24")
        _draw_centered(draw, (cx, y1 + 34), short_detector(detector), label_font, "#1b1f24")
        vals = auc_rows[auc_rows["detector"] == detector]["roc_auc"].astype(float).to_numpy()
        jitter = np.linspace(-38, 38, len(vals)) if len(vals) > 1 else np.array([0.0])
        for offset, val in zip(jitter, vals):
            py = _png_scale(float(val), ymin, ymax, y1, y0)
            draw.ellipse((cx + offset - 7, py - 7, cx + offset + 7, py + 7), fill="#1b1f24")
    draw.text((24, (y0 + y1) // 2), "ROC-AUC", font=label_font, fill="#5d6673")
    draw.multiline_text((150, 940), "LLM-judge is excluded from ROC-AUC because current outputs are binary yes/no verdicts, not continuous scores.", font=label_font, fill="#5d6673")
    out_path = FIGURES_DIR / "detector_comparison_bar.png"
    image.save(out_path, dpi=(200, 200))
    return out_path


def write_lance_png_assets(rag: pd.DataFrame, detection: pd.DataFrame) -> List[Path]:
    outputs: List[Path] = []
    for path in [
        write_embedding_small_multiples_png(),
        write_rag_accuracy_png(rag),
        write_detector_comparison_png(detection),
    ]:
        if path is not None:
            outputs.append(path)
    return outputs


def _llm_stratified_sample(labels: np.ndarray, max_docs: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if max_docs >= len(labels):
        return np.arange(len(labels))
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    pos_frac = len(pos_idx) / max(len(labels), 1)
    n_pos = max(1, int(round(max_docs * pos_frac))) if len(pos_idx) else 0
    n_neg = max_docs - n_pos
    n_pos = min(n_pos, len(pos_idx))
    n_neg = min(n_neg, len(neg_idx))
    sampled_pos = rng.choice(pos_idx, size=n_pos, replace=False) if n_pos else np.array([], dtype=int)
    sampled_neg = rng.choice(neg_idx, size=n_neg, replace=False) if n_neg else np.array([], dtype=int)
    chosen = np.concatenate([sampled_pos, sampled_neg])
    rng.shuffle(chosen)
    return np.sort(chosen)


def aligned_labels_scores(variant: str, detector: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    labels_path = EMBEDDINGS_DIR / f"{variant}_labels.npy"
    score_path = SCORES_DIR / f"{variant}_{detector}_scores.npy"
    if not labels_path.exists() or not score_path.exists():
        return None
    labels = np.load(labels_path)
    scores = np.load(score_path)
    if len(scores) == len(labels):
        return labels.astype(int), scores.astype(float)
    if detector == "neural_classifier":
        indices = np.arange(len(labels))
        _, test_idx = train_test_split(indices, train_size=0.70, random_state=SEED, stratify=labels)
        if len(scores) == len(test_idx):
            return labels[test_idx].astype(int), scores.astype(float)
    if detector == "llm_judge":
        keep = _llm_stratified_sample(labels, len(scores), SEED)
        if len(scores) == len(keep):
            return labels[keep].astype(int), scores.astype(float)
    return None


def write_roc_svg(variant: str, curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]]) -> Path:
    width, height = 720, 600
    x0, y0, x1, y1 = 78, 86, width - 190, height - 74
    body: List[str] = [axis_lines(x0, y0, x1, y1)]
    body.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y0}" stroke="#b8bec5" stroke-dasharray="5 5" stroke-width="1.4"/>')
    for tick in np.linspace(0, 1, 6):
        tx = scale(float(tick), 0, 1, x0, x1)
        ty = scale(float(tick), 0, 1, y1, y0)
        body.append(f'<line x1="{tx:.1f}" y1="{y1}" x2="{tx:.1f}" y2="{y1 + 4}" stroke="#d9ded8"/>')
        body.append(f'<line x1="{x0 - 4}" y1="{ty:.1f}" x2="{x0}" y2="{ty:.1f}" stroke="#d9ded8"/>')
        body.append(f'<text x="{tx:.1f}" y="{y1 + 22}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#5d6673">{tick:.1f}</text>')
        body.append(f'<text x="{x0 - 10}" y="{ty + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#5d6673">{tick:.1f}</text>')

    legend_items: List[Tuple[str, str]] = []
    for detector, (fpr, tpr, auc_value) in curves.items():
        color = COLORS.get(detector, "#277da1")
        points = " ".join(
            f"{scale(float(x), 0, 1, x0, x1):.1f},{scale(float(y), 0, 1, y1, y0):.1f}"
            for x, y in zip(fpr, tpr)
        )
        body.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        legend_items.append((f"{short_detector(detector)} AUC={auc_value:.3f}", color))
    body.append(legend(legend_items, x1 + 22, y0 + 10))
    body.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{height - 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#5d6673">False positive rate</text>')
    body.append(f'<text x="22" y="{(y0 + y1) / 2:.1f}" transform="rotate(-90 22 {(y0 + y1) / 2:.1f})" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#5d6673">True positive rate</text>')
    out_path = FIGURES_DIR / f"roc_curves_{variant}.svg"
    out_path.write_text(
        svg_wrap(width, height, "\n".join(body), f"ROC Curves: {variant}", "Curves use aligned score arrays; sampled/test-split scores are label-aligned where possible."),
        encoding="utf-8",
    )
    return out_path


def write_roc_figures() -> List[Path]:
    outputs: List[Path] = []
    for variant in VARIANTS:
        curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}
        for detector in DETECTORS:
            aligned = aligned_labels_scores(variant, detector)
            if aligned is None:
                continue
            labels, scores = aligned
            if len(np.unique(labels)) < 2 or len(np.unique(scores)) < 2:
                continue
            fpr, tpr, _ = roc_curve(labels, scores)
            curves[detector] = (fpr, tpr, float(sklearn_auc(fpr, tpr)))
        if curves:
            outputs.append(write_roc_svg(variant, curves))
    return outputs


def copy_embedding_figures() -> List[Path]:
    outputs: List[Path] = []
    for variant in VARIANTS:
        src = MODULE2_VIS_DIR / f"embedding_space_{variant}.svg"
        if not src.exists():
            continue
        dst = FIGURES_DIR / f"embedding_space_{variant}.svg"
        shutil.copyfile(src, dst)
        outputs.append(dst)
    preview = MODULE2_VIS_DIR / "embedding_space_all_variants_preview.png"
    if preview.exists():
        dst = FIGURES_DIR / "embedding_space_all_variants_preview.png"
        shutil.copyfile(preview, dst)
        outputs.append(dst)
    return outputs


def percentile(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks / max(len(values) - 1, 1)


def one_line(text: str, limit: int = 260) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def change_excerpt(original: str, poisoned: str, radius: int = 120) -> Tuple[str, str]:
    start = 0
    max_shared = min(len(original), len(poisoned))
    while start < max_shared and original[start] == poisoned[start]:
        start += 1

    end_original = len(original)
    end_poisoned = len(poisoned)
    while (
        end_original > start
        and end_poisoned > start
        and original[end_original - 1] == poisoned[end_poisoned - 1]
    ):
        end_original -= 1
        end_poisoned -= 1

    left = max(0, start - radius)
    right_original = min(len(original), end_original + radius)
    right_poisoned = min(len(poisoned), end_poisoned + radius)
    original_excerpt = ("..." if left > 0 else "") + original[left:right_original] + ("..." if right_original < len(original) else "")
    poisoned_excerpt = ("..." if left > 0 else "") + poisoned[left:right_poisoned] + ("..." if right_poisoned < len(poisoned) else "")
    return one_line(original_excerpt, 360), one_line(poisoned_excerpt, 360)


def load_change_logs() -> Dict[str, Dict[str, Dict[str, str]]]:
    logs: Dict[str, Dict[str, Dict[str, str]]] = {}
    for variant in VARIANTS:
        path = CHANGE_LOG_DIR / f"{variant}_changes.jsonl"
        variant_logs: Dict[str, Dict[str, str]] = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    doc_id = str(rec.get("doc_id", ""))
                    if doc_id:
                        variant_logs[doc_id] = rec
        logs[variant] = variant_logs
    return logs


def markdown_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for col in columns:
            value = str(row.get(col, "")).replace("|", "\\|")
            values.append(value)
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *body])


def before_after_snippet(row: Dict[str, Any], limit: int = 145) -> str:
    before = one_line(row.get("original_excerpt", ""), limit)
    after = one_line(row.get("poisoned_excerpt", ""), limit)
    return f"Before: {before}<br>After: {after}"


def failure_spotlight_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    factual_examples = [
        r
        for r in rows
        if r["attack_type"] == "factual_swap" and r.get("has_change_log_text")
    ][:3]
    semantic_examples = [
        r
        for r in rows
        if r["attack_type"] == "semantic_distortion" and r.get("has_change_log_text")
    ][:3]
    boundary_examples = sorted(
        [
            r
            for r in rows
            if r.get("has_change_log_text")
        ],
        key=lambda r: abs(float(r.get("avg_score_percentile", 0.0)) - poison_rate_from_variant(str(r.get("variant", "")))),
    )[:3]
    near_identical = next(
        (r for r in factual_examples if r["variant"] == "factual_0.1" and r["doc_id"] == "wiki_00095_chunk_018"),
        factual_examples[0] if factual_examples else {},
    )
    natural_distortion = next(
        (r for r in semantic_examples if r["variant"] == "semantic_0.1" and r["doc_id"] == "wiki_00095_chunk_018"),
        semantic_examples[0] if semantic_examples else {},
    )
    boundary_case = boundary_examples[0] if boundary_examples else {}

    spotlight_rows: List[Dict[str, Any]] = []
    if near_identical:
        spotlight_rows.append(
            {
                "category": "Category 1: Hard Factual Swaps",
                "variant": near_identical.get("variant", ""),
                "doc_id": near_identical.get("doc_id", ""),
                "avg_score_percentile": near_identical.get("avg_score_percentile", ""),
                "original_excerpt": near_identical.get("original_excerpt", ""),
                "poisoned_excerpt": near_identical.get("poisoned_excerpt", ""),
                "example": before_after_snippet(near_identical),
                "why_it_fails": "A single date/fact token changes while the surrounding topic, entities, and writing style remain stable, so the embedding barely shifts.",
            }
        )
    if natural_distortion:
        spotlight_rows.append(
            {
                "category": "Category 2: Semantic Overlap",
                "variant": natural_distortion.get("variant", ""),
                "doc_id": natural_distortion.get("doc_id", ""),
                "avg_score_percentile": natural_distortion.get("avg_score_percentile", ""),
                "original_excerpt": natural_distortion.get("original_excerpt", ""),
                "poisoned_excerpt": natural_distortion.get("poisoned_excerpt", ""),
                "example": before_after_snippet(natural_distortion),
                "why_it_fails": "The rewrite is fluent and topical. all-mpnet-base-v2 preserves topic neighborhood more strongly than factual truth, so the passage looks clean.",
            }
        )
    if boundary_case:
        spotlight_rows.append(
            {
                "category": "Category 3: Boundary Cases",
                "variant": boundary_case.get("variant", ""),
                "doc_id": boundary_case.get("doc_id", ""),
                "avg_score_percentile": boundary_case.get("avg_score_percentile", ""),
                "original_excerpt": boundary_case.get("original_excerpt", ""),
                "poisoned_excerpt": boundary_case.get("poisoned_excerpt", ""),
                "example": before_after_snippet(boundary_case),
                "why_it_fails": "The score sits near the expected poison-rate cutoff, so small threshold changes can flip the decision and make mitigation unstable.",
            }
        )
    return spotlight_rows


def read_failure_spotlight_rows() -> List[Dict[str, Any]]:
    path = FAILURE_DIR / "failure_candidates_low_anomaly_poisoned_docs.csv"
    if not path.exists():
        return []
    rows = pd.read_csv(path).to_dict(orient="records")
    for row in rows:
        value = row.get("has_change_log_text", False)
        row["has_change_log_text"] = str(value).lower() == "true" if isinstance(value, str) else bool(value)
    return failure_spotlight_rows(rows)


def failure_spotlight_html() -> str:
    rows = read_failure_spotlight_rows()
    if not rows:
        return "<p>No failure-analysis examples could be generated from the available files.</p>"
    body = []
    for row in rows:
        evidence = (
            f"<strong>{html.escape(str(row.get('variant', '')))}</strong><br>"
            f"<span>{html.escape(str(row.get('doc_id', '')))}</span><br>"
            f"<span>avg anomaly percentile: {html.escape(str(row.get('avg_score_percentile', '')))}</span>"
        )
        before = html.escape(one_line(row.get("original_excerpt", ""), 260))
        after = html.escape(one_line(row.get("poisoned_excerpt", ""), 260))
        body.append(
            f"""
            <tr>
              <td>{html.escape(str(row.get("category", "")))}</td>
              <td>{evidence}</td>
              <td><strong>Before:</strong> {before}<br><strong>After:</strong> {after}</td>
              <td>{html.escape(str(row.get("why_it_fails", "")))}</td>
            </tr>
            """
        )
    return f"""
    <table class="data-table evidence-table">
      <thead><tr><th>Failure Category</th><th>Evidence</th><th>Before / After Text</th><th>Why Detection Failed</th></tr></thead>
      <tbody>{''.join(body)}</tbody>
    </table>
    """


def write_failure_candidates() -> Tuple[Path, Path]:
    rows: List[Dict[str, Any]] = []
    change_logs = load_change_logs()
    for variant in VARIANTS:
        labels_path = EMBEDDINGS_DIR / f"{variant}_labels.npy"
        ids_path = EMBEDDINGS_DIR / f"{variant}_doc_ids.json"
        if not labels_path.exists() or not ids_path.exists():
            continue
        labels = np.load(labels_path).astype(int)
        doc_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        score_arrays: Dict[str, np.ndarray] = {}
        for detector in ["isolation_forest", "lof", "perplexity"]:
            path = SCORES_DIR / f"{variant}_{detector}_scores.npy"
            if path.exists():
                score_arrays[detector] = np.load(path).astype(float)
        if not score_arrays:
            continue

        score_percentiles = {name: percentile(scores) for name, scores in score_arrays.items()}
        avg_percentile = np.mean(np.vstack(list(score_percentiles.values())), axis=0)
        poisoned_idx = np.where(labels == 1)[0]
        hard_idx = sorted(poisoned_idx, key=lambda i: avg_percentile[i])[:25]
        for idx in hard_idx:
            doc_id = doc_ids[int(idx)]
            change = change_logs.get(variant, {}).get(doc_id, {})
            original_excerpt = ""
            poisoned_excerpt = ""
            if change:
                original_excerpt, poisoned_excerpt = change_excerpt(
                    change.get("original_text", ""),
                    change.get("poisoned_text", ""),
                )
            row: Dict[str, Any] = {
                "variant": variant,
                "attack_type": attack_type_from_variant(variant),
                "doc_index": int(idx),
                "doc_id": doc_id,
                "candidate_type": "poisoned_low_anomaly_score",
                "avg_score_percentile": round(float(avg_percentile[idx]), 4),
                "has_change_log_text": bool(change),
                "original_excerpt": original_excerpt,
                "poisoned_excerpt": poisoned_excerpt,
            }
            for detector, scores in score_arrays.items():
                row[f"{detector}_score"] = float(scores[idx])
                row[f"{detector}_percentile"] = round(float(score_percentiles[detector][idx]), 4)
            rows.append(row)

    out_csv = FAILURE_DIR / "failure_candidates_low_anomaly_poisoned_docs.csv"
    if rows:
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        out_csv.write_text("variant,doc_id,note\n,,No candidates could be generated from available files.\n", encoding="utf-8")

    factual_examples = [
        r
        for r in rows
        if r["attack_type"] == "factual_swap" and r.get("has_change_log_text")
    ][:3]
    semantic_examples = [
        r
        for r in rows
        if r["attack_type"] == "semantic_distortion" and r.get("has_change_log_text")
    ][:3]
    factual_table = markdown_table(
        factual_examples,
        ["variant", "doc_id", "avg_score_percentile", "original_excerpt", "poisoned_excerpt"],
    )
    semantic_table = markdown_table(
        semantic_examples,
        ["variant", "doc_id", "avg_score_percentile", "original_excerpt", "poisoned_excerpt"],
    )
    auc_mean = math.nan
    if DETECTION_METRICS.exists():
        det_for_summary = pd.read_csv(DETECTION_METRICS)
        auc_series = pd.to_numeric(det_for_summary.get("roc_auc", pd.Series(dtype=float)), errors="coerce").dropna()
        if not auc_series.empty:
            auc_mean = float(auc_series.mean())
    spotlight_rows = failure_spotlight_rows(rows)
    spotlight_table = markdown_table(
        spotlight_rows,
        ["category", "variant", "doc_id", "example", "why_it_fails"],
    )

    taxonomy = """# Failure Taxonomy Draft

Status: evidence-backed draft. Hardik's poisoned KB files and change logs are available under `data/poisoned_kb/` and `logs/`, so the examples below are tied to concrete poisoned-text edits.

## Quantitative Summary

The failure analysis starts from the quantitative result: the available continuous detector AUCs average **{auc_mean:.3f}**, which is close to random guessing. This means the detectors do not find a stable score ordering where poisoned documents reliably rank above clean documents. The qualitative examples below explain why.

## Concrete Before/After Examples

These examples make the detector failure visible without needing to argue that the code is broken.

{spotlight_table}

## Category 1: Hard Factual Swaps

The factual-swap variants are difficult because the changed facts are often small substitutions inside otherwise topical, fluent Wikipedia-style passages. The surrounding context stays the same, so the embedding remains near the original semantic neighborhood. In the current outputs, embedding-based detector AUCs stay near random chance, and the embedding projections place poisoned points inside the same local clusters as clean points.

Representative low-anomaly factual-swap candidates:

{factual_table}

## Category 2: Semantic Overlap

The semantic-distortion variants rewrite passages more broadly, but they preserve topic, genre, and many named entities. That makes them look distributionally clean to the embedding model. The poisoned documents still do not form a separable region in the embedding projection, supporting the finding that all-mpnet-base-v2 embeddings are relatively invariant to these stealthy perturbations.

Representative low-anomaly semantic-distortion candidates:

{semantic_table}

## Category 3: Boundary Cases

Boundary cases are poisoned documents whose detector scores fall close to the practical cutoff used to decide what gets removed. Because clean and poisoned score distributions overlap, small changes to the threshold can either let poison through or remove too many clean documents. The unsupervised detectors flag almost nothing under automatic thresholds, while the neural classifier can over-flag badly.

This directly explains the `factual_0.2` threshold-mitigation crisis: the filter could not find a clean boundary and flagged the whole KB, leaving no documents for vector-store creation.

## Impact Statement

These failures directly caused the mitigation crisis. When detectors cannot separate poison from truth, mitigation becomes a thresholding problem: conservative thresholds miss poison, while aggressive thresholds remove clean evidence. In `factual_0.2`, naive threshold filtering removed every document and emptied the KB. The capped top-K mitigation fixes that engineering failure by preserving a usable KB across all five variants, but the end-to-end RAG score remains essentially neutral versus undefended retrieval.

## Final Interpretation

The failed cases are not random noise. They are exactly the cases expected to defeat embedding-space defenses: small factual substitutions or fluent semantic rewrites that keep the passage anchored to the same topic. The right report narrative is therefore the Stealthy Poisoning Paradox: the attack is damaging because it changes truth, but it is hard to detect because it preserves topic and style.
""".format(
        auc_mean=auc_mean,
        factual_table=factual_table,
        semantic_table=semantic_table,
        spotlight_table=spotlight_table,
    )
    out_md = FAILURE_DIR / "failure_taxonomy_draft.md"
    out_md.write_text(taxonomy, encoding="utf-8")
    return out_csv, out_md


def write_coverage_report(figure_paths: Sequence[Path], table_paths: Dict[str, Path]) -> Path:
    rel_figures = [str(path.relative_to(REPO_ROOT)).replace("\\", "/") for path in figure_paths]
    rel_tables = {k: str(v.relative_to(REPO_ROOT)).replace("\\", "/") for k, v in table_paths.items()}
    report = f"""# Final Visualization Coverage

Generated by `python scripts/run_module4_dashboard.py`.

## Results Dashboard

Status: covered as a final presentable dashboard.

- Notebook: `notebooks/results/dashboard.ipynb`
- Standalone HTML report: `results/module4_visualization_report.html`
- Final integration notes: `results/module4_handoff_action_items.md`
- Final visualization package: `results/module4_lance_handoff_package.md`
- Table 1 RAG accuracy: `{rel_tables.get("rag_accuracy", "")}`
- Table 2 detector performance: `{rel_tables.get("detector_performance", "")}`
- Table 3 mitigation filter summary: `{rel_tables.get("mitigation_summary", "")}`

Caveat: `results/metrics.csv` now contains threshold-mitigation rows plus top-K capped mitigation rows. The top-K rows cover all five variants; the old threshold path still documents the `factual_0.2` empty-KB failure.

## Embedding Space Visualization

Status: covered with t-SNE, and ready for an optional UMAP rerun if `umap-learn` is installed.

- Interactive dashboard: `Module 2/results/visualizations/embedding_space_dashboard.html`
- Report figures copied into `results/figures/`: `{", ".join(p for p in rel_figures if "embedding_space_" in p) or "missing"}`
- Lance report PNG: `results/figures/umap_small_multiples.png`

Finding: poisoned points overlap clean points across factual and semantic variants, matching the near-random detector AUCs.

## Performance Curves

Status: covered for the available final data.

- Detector F1/AUC figure: `results/figures/detector_f1_auc_by_variant.svg`
- Precision/recall/F1 figure: `results/figures/detector_precision_recall_f1_10_30.svg`
- Mitigation filter precision/recall figure: `results/figures/mitigation_filter_precision_recall.svg`
- ROC figures: `results/figures/roc_curves_<variant>.svg`
- RAG accuracy figure: `results/figures/rag_accuracy_vs_contamination_available.svg`
- Lance report PNGs: `results/figures/rag_accuracy_curves.png`, `results/figures/detector_comparison_bar.png`

Caveat: the RAG strategy plot uses the available n=5 recovery rows. Top-K capped mitigation covers all five variants; threshold mitigation remains an old comparison case and is missing `factual_0.2` because that filtered KB was empty.

## Error Analysis & Failure Taxonomy

Status: covered as an evidence-backed taxonomy, with optional manual refinement.

- Candidate low-anomaly poisoned docs: `results/failure_analysis/failure_candidates_low_anomaly_poisoned_docs.csv`
- Evidence-backed taxonomy draft: `results/failure_analysis/failure_taxonomy_draft.md`

Caveat: Hardik's poisoned KB JSONL files and change logs are now present locally, so the taxonomy includes concrete examples. It can still be strengthened by manually reading more sampled cases.

## Final Integration Notes

- All five poisoned variants are represented in the embedding figures and top-K RAG strategy comparison.
- The threshold `factual_0.2` empty-KB case is kept as evidence that naive mitigation can break the system.
- The final narrative should be: naive defense hurts, safe top-K defense is stable but neutral, and the root issue is embedding-space overlap.

Additional caveat: LLM-judge AUC is not available because the current judge outputs binary yes/no verdicts rather than continuous scores for ROC ranking.
"""
    COVERAGE_REPORT.write_text(report, encoding="utf-8")
    return COVERAGE_REPORT


def write_handoff_notes() -> Path:
    notes = """# Final Integration Notes

## Conceptual Pivot: Stealthy Poisoning Paradox

The visualizations are technically correct, but they overturn the original hypothesis. The poisoned documents do not form distinct clusters in embedding space. Instead, poisoned points overlap clean points across factual-swap and semantic-distortion variants. This is the Stealthy Poisoning Paradox: the attack changes truth enough to hurt RAG, but preserves topic and style enough to evade embedding-space defenses.

## What Is Covered

- Embedding-space figures for all five variants.
- Presentation PNGs: `umap_small_multiples.png`, `rag_accuracy_curves.png`, `detector_comparison_bar.png`.
- Detector precision/recall/F1/AUC table with 25 rows.
- Detector F1/AUC bar chart.
- Precision/recall/F1 chart for 10% and 30% contamination.
- Mitigation filter precision/recall chart.
- ROC diagnostics for variants with aligned score arrays.
- RAG strategy comparison with undefended, threshold, and top-K capped rows.
- Failure-candidate CSV based on poisoned documents with low anomaly scores.
- Evidence-backed failure-taxonomy draft using Hardik's change logs.

## Final Narrative

Naive threshold mitigation is the destructive comparison case: it can remove too much clean evidence and it collapses `factual_0.2` by emptying the KB. Top-K capped mitigation is the safe engineering fix: all five variants now run without KB collapse. The remaining research finding is that safe mitigation is neutral rather than helpful because the detector still cannot cleanly separate poison from truth.

## Caveats To Keep In The Report

- LLM-judge AUC is unavailable because the current judge produces binary yes/no outputs, not continuous scores.
- RAG strategy results are n=5 recovery rows, so they are best used as directional evidence for the mitigation trade-off.
- Threshold `factual_0.2` is missing because the old filter removed all documents; top-K capped `factual_0.2` is present and stable.
- The failure taxonomy now includes concrete examples from Hardik's change logs, but it can still be refined by manual review.
- The current generated embedding projections are t-SNE. UMAP can be rerun later if `umap-learn` is installed, but the current conclusion already matches the detector metrics.
- RAG EM/token-F1 normalization is handled in `scripts/evaluate_rag.py`: lowercase, punctuation removal, article removal, and whitespace normalization before EM/F1.
"""
    HANDOFF_PATH.write_text(notes, encoding="utf-8")
    return HANDOFF_PATH


def write_lance_handoff_package(table_paths: Dict[str, Path]) -> Path:
    rag, detection, mitigation = read_tables()
    rag = latest_rag_rows(rag)
    summary = report_summary(rag, detection, mitigation)
    rel_tables = {
        key: str(path.relative_to(REPO_ROOT)).replace("\\", "/")
        for key, path in table_paths.items()
    }
    lance = f"""# Final Visualization Package

## Narrative To Use

Frame the result as the **Stealthy Poisoning Paradox**:

1. **Overlap finding:** high-quality factual swaps and semantic distortions do not create a clean anomaly cluster in embedding space. Detector AUCs average **{fmt(summary["auc_mean"], 3)}**, close to random guessing.
2. **Naive mitigation crisis:** the old threshold strategy can remove too much clean evidence. `factual_0.2` is the clearest warning case because threshold filtering emptied the KB.
3. **Safe mitigation result:** top-K capped mitigation now covers all five variants and preserves KB integrity. It is stable, but it is neutral rather than performance-improving because token-F1 matches undefended retrieval at about **{fmt(summary["topk_f1_mean"], 4)}**.

## Visual Assets

Use these files directly from `results/figures/`:

| File | Use in report |
|---|---|
| `results/figures/umap_small_multiples.png` | Five-variant embedding overlap figure. Filename is report-friendly; local projection is t-SNE because `umap-learn` is not installed. |
| `results/figures/rag_accuracy_curves.png` | RAG strategy figure comparing undefended, naive threshold, and top-K capped mitigation. |
| `results/figures/detector_comparison_bar.png` | AUC comparison figure with the random-guess line at 0.5. |

Supporting tables:

| Table | Path |
|---|---|
| RAG accuracy/recovery | `{rel_tables.get("rag_accuracy", "")}` |
| Detector performance | `{rel_tables.get("detector_performance", "")}` |
| Mitigation summary | `{rel_tables.get("mitigation_summary", "")}` |

## Failure Taxonomy Categories

- **Category 1: Hard Factual Swaps:** one date/name/fact changes while the surrounding passage stays almost identical, causing minimal embedding shift.
- **Category 2: Semantic Overlap:** the rewrite is fluent and topical, so all-mpnet-base-v2 treats it like a natural paraphrase instead of a threat.
- **Category 3: Boundary Cases:** scores fall near the practical cutoff, so threshold changes either miss poison or remove too much clean evidence.

Detailed before/after examples from Hardik's change logs are in `results/failure_analysis/failure_taxonomy_draft.md`.

## Final Technical Check

- Report PNG assets are generated at 200 DPI.
- EM/F1 normalization is centralized in `scripts/evaluate_rag.py`: lowercase, punctuation removal, article removal, and whitespace normalization before scoring.
- Threshold `factual_0.2` is not a missing visualization; it is a real mitigation failure caused by an empty filtered KB. Top-K capped `factual_0.2` is now present.

## Contribution Table

| Contributor | Contribution |
|---|---|
| Latika | Final visualization dashboard, embedding-space overlap analysis, detector/mitigation/RAG performance plots, error analysis, and failure taxonomy. |
| Anushree | Detection metrics, embeddings, detector scores, mitigation summary, and top-K capped mitigation rows. |
| Hardik | Poisoned KB variants, labels, and change logs used for text-backed taxonomy examples. |
| Vatsal | RAG baseline, undefended, threshold-mitigated, and top-K capped recovery metrics. |
| Lance | Final report integration and narrative synthesis. |
"""
    LANCE_HANDOFF_PATH.write_text(lance, encoding="utf-8")
    return LANCE_HANDOFF_PATH


def latest_clean_metrics(rag: pd.DataFrame) -> Dict[str, float]:
    if rag.empty:
        return {"n_examples": math.nan, "exact_match": math.nan, "token_f1": math.nan}
    clean = rag[rag.get("attack_type", "") == "clean"].copy()
    if clean.empty:
        clean = rag.copy()
    clean["n_examples"] = pd.to_numeric(clean["n_examples"], errors="coerce")
    row = clean.sort_values("n_examples").iloc[-1]
    return {
        "n_examples": float(row.get("n_examples", math.nan)),
        "exact_match": float(row.get("exact_match", math.nan)),
        "token_f1": float(row.get("token_f1", math.nan)),
    }


def report_summary(rag: pd.DataFrame, detection: pd.DataFrame, mitigation: pd.DataFrame) -> Dict[str, Any]:
    rag = latest_rag_rows(rag)
    clean = latest_clean_metrics(rag)
    aucs = detection["roc_auc"].dropna() if "roc_auc" in detection else pd.Series(dtype=float)
    f1s = detection["f1"].dropna() if "f1" in detection else pd.Series(dtype=float)
    best_f1 = detection.sort_values("f1", ascending=False).head(1) if not detection.empty else pd.DataFrame()
    recovery = rag[rag.get("attack_type", "") != "clean"].copy() if not rag.empty else pd.DataFrame()
    mitigated = recovery[recovery.get("detector", "") == "mitigated"].copy() if not recovery.empty else pd.DataFrame()
    topk = recovery[recovery.get("detector", "") == "mitigated_topK"].copy() if not recovery.empty else pd.DataFrame()
    undefended = recovery[recovery.get("detector", "") == "none"].copy() if not recovery.empty else pd.DataFrame()
    recovery_variants = 0
    if not recovery.empty:
        recovery_variants = int(len(recovery[["attack_type", "poison_rate"]].drop_duplicates()))
    return {
        "clean_n": int(clean["n_examples"]) if math.isfinite(clean["n_examples"]) else 0,
        "clean_em": clean["exact_match"],
        "clean_f1": clean["token_f1"],
        "recovery_rows": int(len(recovery)),
        "recovery_variants": recovery_variants,
        "mitigated_rows": int(len(mitigated)),
        "topk_rows": int(len(topk)),
        "undefended_rows": int(len(undefended)),
        "mitigated_f1_min": float(mitigated["token_f1"].min()) if not mitigated.empty else math.nan,
        "mitigated_f1_max": float(mitigated["token_f1"].max()) if not mitigated.empty else math.nan,
        "topk_f1_mean": float(topk["token_f1"].mean()) if not topk.empty else math.nan,
        "undefended_f1_mean": float(undefended["token_f1"].mean()) if not undefended.empty else math.nan,
        "detector_rows": int(len(detection)),
        "auc_min": float(aucs.min()) if not aucs.empty else math.nan,
        "auc_max": float(aucs.max()) if not aucs.empty else math.nan,
        "auc_mean": float(aucs.mean()) if not aucs.empty else math.nan,
        "f1_max": float(f1s.max()) if not f1s.empty else math.nan,
        "best_detector": "" if best_f1.empty else str(best_f1.iloc[0]["detector"]),
        "best_variant": "" if best_f1.empty else str(best_f1.iloc[0]["variant"]),
        "mitigation_rows": int(len(mitigation)),
        "filter_recall_min": float(mitigation["filter_recall"].min()) if not mitigation.empty else math.nan,
        "filter_recall_max": float(mitigation["filter_recall"].max()) if not mitigation.empty else math.nan,
        "filter_precision_min": float(mitigation["filter_precision"].min()) if not mitigation.empty else math.nan,
        "filter_precision_max": float(mitigation["filter_precision"].max()) if not mitigation.empty else math.nan,
    }


def fmt(value: Any, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def figure_info(rel_path: str) -> Dict[str, str]:
    name = Path(rel_path).name
    if name == "embedding_space_all_variants_preview.png":
        return {
            "section": "Embedding Space",
            "title": "Embedding Space Overview",
            "caption": (
                "Small-multiple view of all five poisoning variants. Orange poisoned documents remain embedded "
                "inside the same local neighborhoods as clean documents, which is the central visual result."
            ),
        }
    if name == "umap_small_multiples.png":
        return {
            "section": "Embedding Space",
            "title": "Report Asset: Embedding Small Multiples",
            "caption": (
                "Presentation PNG showing all five variants. The local projection is t-SNE because UMAP is not "
                "installed, but the visual finding is the required overlap pattern: poisoned points stay inside "
                "clean neighborhoods."
            ),
        }
    if name.startswith("embedding_space_"):
        variant = name.removeprefix("embedding_space_").removesuffix(".svg")
        return {
            "section": "Embedding Space",
            "title": f"Embedding Projection: {variant}",
            "caption": (
                "Gray points are clean reference documents, blue points are clean documents in the poisoned KB, "
                "and orange points are true poisoned documents. The lack of separation supports the finding that "
                "embedding-space anomaly detection is weak for these attacks."
            ),
        }
    if name == "rag_accuracy_vs_contamination_available.svg":
        return {
            "section": "RAG Accuracy",
            "title": "RAG Strategy Comparison",
            "caption": (
                "The final RAG comparison separates the old threshold filter from the safer top-K capped filter. "
                "Threshold mitigation can collapse the KB; top-K capped mitigation keeps the system usable but "
                "does not improve token-F1 over undefended retrieval."
            ),
        }
    if name == "rag_accuracy_curves.png":
        return {
            "section": "RAG Accuracy",
            "title": "Report Asset: RAG Strategy Comparison",
            "caption": (
                "Presentation PNG comparing undefended retrieval, naive threshold mitigation, and capped top-K "
                "mitigation. The safe top-K fix prevents KB collapse but remains neutral versus undefended F1."
            ),
        }
    if name == "detector_f1_auc_by_variant.svg":
        return {
            "section": "Detector Performance",
            "title": "Detector F1 and AUC by Variant",
            "caption": (
                "Detector AUCs cluster near 0.5, so the detectors are close to random ranking in embedding/score space. "
                "F1 values should be interpreted together with flagging behavior because some methods over-flag."
            ),
        }
    if name == "detector_comparison_bar.png":
        return {
            "section": "Detector Performance",
            "title": "Report Asset: Detector AUC Comparison",
            "caption": (
                "Presentation PNG showing detector AUCs clustered around the random-guess line. This supports the "
                "Stealthy Poisoning Paradox narrative."
            ),
        }
    if name == "detector_precision_recall_f1_10_30.svg":
        return {
            "section": "Detector Performance",
            "title": "Precision, Recall, and F1 at Low/High Poisoning",
            "caption": (
                "This chart compares detector tradeoffs at 10% and 30% contamination. The neural model often gets high "
                "recall by flagging broadly, while unsupervised methods miss most poisoned documents under automatic thresholds."
            ),
        }
    if name == "mitigation_filter_precision_recall.svg":
        return {
            "section": "Mitigation",
            "title": "Mitigation Filter Precision and Recall",
            "caption": (
                "The filter can remove many poisoned documents, but precision is low because many clean documents are dropped too. "
                "This explains why defended RAG recovery must be evaluated separately before claiming end-to-end improvement."
            ),
        }
    if name.startswith("roc_curves_"):
        variant = name.removeprefix("roc_curves_").removesuffix(".svg")
        return {
            "section": "ROC Diagnostics",
            "title": f"ROC Curves: {variant}",
            "caption": (
                "ROC curves near the diagonal indicate weak score separation between poisoned and clean documents. "
                "This is consistent with the embedding-projection overlap and near-random AUC values."
            ),
        }
    return {"section": "Other", "title": name, "caption": ""}


def report_intro_markdown(summary: Dict[str, Any]) -> str:
    return f"""# Final Visualization Report: Stealthy Poisoning in RAG

**Project question.** This project studies whether poisoned documents in a RAG knowledge base can be detected and mitigated before they degrade question-answering behavior.

**Visualization goal.** The original hypothesis was that poisoned documents would move into a distinct region of embedding space. The visual evidence says the opposite: poisoned and clean documents strongly overlap. That overlap is the main finding, not a dashboard bug.

**Conceptual pivot.** Frame this as the **Stealthy Poisoning Paradox**: the attacks change factual truth enough to hurt RAG, but preserve topic and writing style enough to remain embedded with clean documents.

## Executive Summary

- Evaluated **5 poisoned KB variants**: factual swaps at 10%, 20%, and 30%; semantic distortions at 10% and 30%.
- Detection metrics table contains **{summary["detector_rows"]} detector rows** across Isolation Forest, LOF, neural classifier, GPT-2 perplexity, and LLM-judge rows where available.
- Available detector AUCs range from **{fmt(summary["auc_min"], 4)}** to **{fmt(summary["auc_max"], 4)}** with mean **{fmt(summary["auc_mean"], 4)}**, which is close to random chance.
- Best observed detector F1 is **{fmt(summary["f1_max"], 4)}** for `{summary["best_detector"]}` on `{summary["best_variant"]}`, but this must be read with precision/recall because broad flagging inflates recall.
- Latest clean RAG baseline uses **{summary["clean_n"]} examples** with EM **{fmt(summary["clean_em"], 4)}** and token-F1 **{fmt(summary["clean_f1"], 4)}**.
- Final RAG rows compare **undefended**, **naive threshold**, and **top-K capped** strategies across **{summary["recovery_variants"]} variant settings**.
- Naive threshold mitigation ranges from token-F1 **{fmt(summary["mitigated_f1_min"], 4)}** to **{fmt(summary["mitigated_f1_max"], 4)}** and still documents the `factual_0.2` empty-KB failure.
- Top-K capped mitigation covers all five variants with mean token-F1 **{fmt(summary["topk_f1_mean"], 4)}**, matching the available undefended baseline **{fmt(summary["undefended_f1_mean"], 4)}** rather than improving it.
- Mitigation filter recall ranges from **{fmt(summary["filter_recall_min"], 3)}** to **{fmt(summary["filter_recall_max"], 3)}**, while precision ranges from **{fmt(summary["filter_precision_min"], 3)}** to **{fmt(summary["filter_precision_max"], 3)}**.
- Mitigation trade-off: naive threshold defense can be destructive, while the safer capped defense preserves the KB but is performance-neutral.
- LLM-judge AUC is intentionally unavailable because current judge outputs are binary yes/no verdicts, not continuous ROC scores.

## How To Read This Report

Orange points represent true poisoned documents. Blue points represent clean documents inside the poisoned KB. Gray points are clean reference documents. Good separation would mean orange points forming their own cluster or region. Instead, they overlap with blue and gray points, indicating that the embedding representation is largely invariant to these poisoning edits.
"""


def markdown_cell(text: str) -> Dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.splitlines()]}


def code_cell(source: Sequence[str]) -> Dict[str, Any]:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": list(source)}


def write_notebook(figure_paths: Sequence[Path], table_paths: Dict[str, Path]) -> Path:
    rel_figures = [str(path.relative_to(REPO_ROOT)).replace("\\", "/") for path in figure_paths]
    rel_tables = {k: str(v.relative_to(REPO_ROOT)).replace("\\", "/") for k, v in table_paths.items()}
    rag, detection, mitigation = read_tables()
    rag = latest_rag_rows(rag)
    summary = report_summary(rag, detection, mitigation)
    grouped: Dict[str, List[str]] = {}
    for fig in rel_figures:
        grouped.setdefault(figure_info(fig)["section"], []).append(fig)
    failure_table = markdown_table(
        read_failure_spotlight_rows(),
        ["category", "variant", "doc_id", "example", "why_it_fails"],
    ) or "No failure-analysis examples could be generated from the available files."

    cells: List[Dict[str, Any]] = [
        markdown_cell(report_intro_markdown(summary)),
        code_cell(
            [
                "from pathlib import Path\n",
                "import pandas as pd\n",
                "from IPython.display import display, SVG, Image, Markdown\n",
                "\n",
                "ROOT = Path.cwd()\n",
                "while not (ROOT / 'results').exists() and ROOT != ROOT.parent:\n",
                "    ROOT = ROOT.parent\n",
                "\n",
                "rag = pd.read_csv(ROOT / 'results' / 'tables' / 'module4_table1_rag_accuracy.csv')\n",
                "det = pd.read_csv(ROOT / 'results' / 'tables' / 'module4_table2_detector_performance.csv')\n",
                "mit = pd.read_csv(ROOT / 'results' / 'tables' / 'module4_table3_mitigation_filter_summary.csv')\n",
            ]
        ),
        markdown_cell(
            """## Results Tables

The tables below are the report source of truth. Table 1 records the final RAG evaluation rows, Table 2 records detector-level precision/recall/F1/AUC, and Table 3 compares threshold versus top-K mitigation behavior.
"""
        ),
        markdown_cell(
            """### Table 1: RAG Accuracy

This table includes the clean baseline, the available poisoned undefended row, the old threshold-mitigated rows, and the new top-K capped rows for all five poisoning variants.
"""
        ),
        code_cell(["display(rag)\n"]),
        markdown_cell(
            """### Table 2: Detector Performance

This table is the main quantitative evidence for the detection layer. AUC values near 0.5 mean the detector score ranking is close to random; F1 values must be interpreted with `n_flagged` because some detectors get recall by flagging many documents.
"""
        ),
        code_cell(["display(det)\n"]),
        markdown_cell(
            """### Table 3: Mitigation Filter Summary

This table makes the mitigation trade-off explicit. The threshold strategy shows the old high-risk filter behavior; the top-K capped rows show the safer strategy that keeps the KB usable.
"""
        ),
        code_cell(["display(mit)\n"]),
    ]

    section_notes = {
        "Embedding Space": (
            "## Embedding Space Visualizations\n\n"
            "These are the most important project figures. The expected success pattern would be orange poisoned points separating from clean points. "
            "The actual pattern is heavy overlap, which explains why embedding-space detectors and score-based detectors struggle."
        ),
        "RAG Accuracy": (
            "## RAG Accuracy Context\n\n"
            "This figure compares undefended retrieval, naive threshold mitigation, and capped top-K mitigation. The final story is not that defense wins; it is that naive defense can break the KB, while safe top-K defense is stable but neutral."
        ),
        "Detector Performance": (
            "## Detector Performance Curves\n\n"
            "These figures summarize whether the detectors can separate poisoned documents from clean documents. The main result is weak separation: most AUCs are close to chance."
        ),
        "Mitigation": (
            "## Mitigation Filter Behavior\n\n"
            "Mitigation is evaluated here as a filtering step. The filter removes many poisoned examples, but it also removes clean documents, so end-to-end RAG recovery remains a separate required measurement."
        ),
        "ROC Diagnostics": (
            "## ROC Diagnostics\n\n"
            "ROC curves give a threshold-independent view of detector quality. Curves near the diagonal reinforce the embedding-overlap finding."
        ),
    }
    for section in ["Embedding Space", "RAG Accuracy", "Detector Performance", "Mitigation", "ROC Diagnostics"]:
        figs = grouped.get(section, [])
        if not figs:
            continue
        cells.append(markdown_cell(section_notes[section]))
        for fig in figs:
            info = figure_info(fig)
            cells.append(markdown_cell(f"### {info['title']}\n\n{info['caption']}"))
            cells.append(
                code_cell(
                    [
                        f"fig = {json.dumps(fig)}\n",
                        "path = ROOT / fig\n",
                        "if path.suffix.lower() == '.svg':\n",
                        "    display(SVG(filename=str(path)))\n",
                        "elif path.suffix.lower() == '.png':\n",
                        "    display(Image(filename=str(path)))\n",
                        "else:\n",
                        "    display(Markdown(f'Unsupported figure type: `{fig}`'))\n",
                    ]
                )
            )

    cells.extend(
        [
            markdown_cell(
                """## Final Integration Notes

All required visualization inputs are now present for the final story: embedding projections, detector metrics, mitigation summaries, poisoned KB change logs, and RAG strategy rows. The threshold `factual_0.2` failure is retained as evidence of naive mitigation risk, while top-K capped `factual_0.2` shows the safer fix is stable.

Additional caveat: LLM-judge AUC is unavailable because the current judge produces binary yes/no verdicts rather than continuous scores for ROC ranking.
"""
            ),
            markdown_cell(
                f"""## Failure Analysis

The table below embeds the key before/after examples directly from the poisoned change logs. These true poisoned documents received low anomaly scores, so the detectors treated them as especially clean-like.

{failure_table}
"""
            ),
            markdown_cell(
                """## Final Takeaway

The final visualization story is coherent: poisoned documents do not separate in mpnet embedding space, detector AUCs stay close to random chance, naive thresholding can damage or empty the KB, and capped top-K mitigation prevents collapse without improving answer quality. This supports the Stealthy Poisoning Paradox: the attack changes truth, but preserves enough topic and style to evade embedding-space defenses.
"""
            ),
        ]
    )

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "module4_table_paths": rel_tables,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    return NOTEBOOK_PATH


def inline_figure_html(path: Path) -> str:
    if path.suffix.lower() == ".svg":
        return path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".png":
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'<img src="data:image/png;base64,{encoded}" alt="{html.escape(path.name)}">'
    return f"<p>Unsupported figure type: {html.escape(path.name)}</p>"


def write_html_report(
    rag: pd.DataFrame,
    detection: pd.DataFrame,
    mitigation: pd.DataFrame,
    figure_paths: Sequence[Path],
) -> Path:
    rag = latest_rag_rows(rag)
    summary = report_summary(rag, detection, mitigation)
    grouped: Dict[str, List[Path]] = {}
    for path in figure_paths:
        rel = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
        grouped.setdefault(figure_info(rel)["section"], []).append(path)

    table_style = 'class="data-table"'
    rag_cols = [
        "timestamp_utc",
        "attack_type",
        "poison_rate",
        "detector",
        "dataset",
        "n_examples",
        "exact_match",
        "token_f1",
        "generation_backend",
        "ollama_model",
    ]
    rag_display = rag[[c for c in rag_cols if c in rag.columns]].copy() if not rag.empty else pd.DataFrame()
    rag_html = rag_display.to_html(index=False, classes="data-table", border=0) if not rag_display.empty else "<p>No RAG rows found.</p>"
    det_html = detection.to_html(index=False, classes="data-table", border=0) if not detection.empty else "<p>No detector rows found.</p>"
    mitigation_display = mitigation_strategy_summary(rag, mitigation)
    mit_html = mitigation_display.to_html(index=False, classes="data-table", border=0) if not mitigation_display.empty else "<p>No mitigation rows found.</p>"

    section_order = ["Embedding Space", "RAG Accuracy", "Detector Performance", "Mitigation", "ROC Diagnostics"]
    section_intro = {
        "Embedding Space": "The embedding projections are the central figure set. Poisoned examples should separate if Hypothesis 1 holds; instead, they overlap clean examples.",
        "RAG Accuracy": "This compares undefended retrieval, naive threshold mitigation, and capped top-K mitigation. Naive defense can break the KB; top-K defense is stable but neutral.",
        "Detector Performance": "These plots summarize score-level detector behavior across variants and contamination rates.",
        "Mitigation": "These curves show why defended RAG recovery can fail: the threshold filter has high recall but low precision, so it can remove useful clean evidence.",
        "ROC Diagnostics": "ROC curves provide threshold-independent evidence that detector scores have weak separation.",
    }
    figure_sections: List[str] = []
    for section in section_order:
        paths = grouped.get(section, [])
        if not paths:
            continue
        cards: List[str] = []
        for path in paths:
            rel = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
            info = figure_info(rel)
            cards.append(
                f"""
                <article class="figure-card">
                  <h3>{html.escape(info["title"])}</h3>
                  <p>{html.escape(info["caption"])}</p>
                  <div class="figure-wrap">{inline_figure_html(path)}</div>
                </article>
                """
            )
        figure_sections.append(
            f"""
            <section>
              <h2>{html.escape(section)}</h2>
              <p>{html.escape(section_intro[section])}</p>
              {''.join(cards)}
            </section>
            """
        )

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Final Visualization Report: Stealthy Poisoning in RAG</title>
  <style>
    :root {{
      --bg: #f7f7f4;
      --panel: #ffffff;
      --text: #1b1f24;
      --muted: #5d6673;
      --line: #d9ded8;
      --accent: #277da1;
      --warn: #d55e00;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }}
    header {{
      padding: 34px 40px 24px;
      background: #fbfbf8;
      border-bottom: 1px solid var(--line);
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 28px 56px; }}
    h1 {{ margin: 0; font-size: 34px; letter-spacing: 0; }}
    h2 {{ margin: 42px 0 10px; font-size: 24px; }}
    h3 {{ margin: 0 0 8px; font-size: 18px; }}
    p {{ color: var(--muted); margin: 8px 0 14px; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 12px;
      margin-top: 22px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .metric strong {{ display: block; font-size: 22px; color: var(--text); }}
    .metric span {{ color: var(--muted); font-size: 13px; }}
    section {{
      margin-top: 28px;
      padding-top: 6px;
    }}
    .figure-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      margin: 16px 0;
      overflow: hidden;
    }}
    .figure-wrap {{
      margin-top: 14px;
      overflow: auto;
      border-top: 1px solid var(--line);
      padding-top: 12px;
    }}
    .figure-wrap svg, .figure-wrap img {{
      max-width: 100%;
      height: auto;
      display: block;
      margin: 0 auto;
    }}
    .table-block {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: auto;
      margin: 14px 0 24px;
    }}
    table.data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    table.data-table th, table.data-table td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      white-space: nowrap;
    }}
    table.data-table th {{ background: #fafaf7; color: var(--muted); }}
    table.evidence-table td {{
      white-space: normal;
      vertical-align: top;
      min-width: 160px;
    }}
    table.evidence-table td:nth-child(3) {{
      min-width: 360px;
      line-height: 1.45;
    }}
    .takeaway {{
      border-left: 4px solid var(--warn);
      background: #fffaf6;
      padding: 14px 16px;
      margin-top: 20px;
      border-radius: 0 8px 8px 0;
    }}
    code {{ background: #ecefeb; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
  <header>
    <h1>Final Visualization Report: Stealthy Poisoning in RAG</h1>
    <p>Evaluation and visualization for detecting poisoned documents in a RAG knowledge base.</p>
    <div class="summary">
      <div class="metric"><strong>5</strong><span>poisoned KB variants</span></div>
      <div class="metric"><strong>{summary["detector_rows"]}</strong><span>detector metric rows</span></div>
      <div class="metric"><strong>{fmt(summary["auc_mean"], 3)}</strong><span>mean detector AUC</span></div>
      <div class="metric"><strong>{fmt(summary["clean_em"], 3)}</strong><span>clean RAG EM</span></div>
      <div class="metric"><strong>{summary["topk_rows"]}</strong><span>top-K capped rows</span></div>
    </div>
  </header>
  <main>
    <section>
      <h2>Executive Summary</h2>
      <p>The original embedding-space hypothesis expected poisoned documents to cluster away from clean documents. The visualizations show the opposite: poisoned documents overlap with clean documents across factual-swap and semantic-distortion variants.</p>
      <div class="takeaway">
        <strong>Stealthy Poisoning Paradox:</strong> the attack changes truth enough to hurt RAG, but preserves topic and writing style enough to stay embedded with clean documents. Mean detector AUC is {fmt(summary["auc_mean"], 3)}, close to random guessing.
      </div>
      <p><strong>Naive mitigation crisis:</strong> the old threshold filter can remove too much clean evidence. <code>factual_0.2</code> is the clearest warning case because threshold filtering emptied the KB.</p>
      <p><strong>Safe mitigation result:</strong> capped top-K mitigation now runs for all five variants and preserves KB integrity, but token-F1 stays neutral at about {fmt(summary["topk_f1_mean"], 4)}, matching the available undefended baseline.</p>
      <p><strong>LLM-judge caveat:</strong> LLM-judge AUC is not shown because the available judge outputs are binary yes/no verdicts, not continuous scores suitable for ROC ranking.</p>
    </section>
    <section>
      <h2>Source Tables</h2>
      <h3>Table 1: RAG Accuracy</h3>
      <p>The final RAG table includes the clean baseline, the available poisoned undefended row, the old threshold-mitigated rows, and the new top-K capped rows for all five poisoning variants.</p>
      <div class="table-block">{rag_html}</div>
      <h3>Table 2: Detector Performance</h3>
      <p>Detector-level precision, recall, F1, and AUC. This is the quantitative companion to the embedding overlap plots.</p>
      <div class="table-block">{det_html}</div>
      <h3>Table 3: Mitigation Filter Summary</h3>
      <p>Mitigation strategy summary. Threshold rows preserve the old high-risk filter behavior; top-K rows show the safer strategy that keeps the KB usable.</p>
      <div class="table-block">{mit_html}</div>
    </section>
    {''.join(figure_sections)}
    <section>
      <h2>Failure Analysis</h2>
      <p>The table below embeds the key before/after examples directly from the poisoned change logs, organized into hard factual swaps, semantic overlap, and boundary cases.</p>
      <div class="table-block">{failure_spotlight_html()}</div>
      <p><strong>Impact:</strong> these categories explain why stealthy poisoning is the primary result. The detectors could not find a clean threshold, which caused the empty-KB failure under naive mitigation and explains why the safer top-K strategy is stable but not performance-improving.</p>
    </section>
    <section>
      <h2>Contribution Summary</h2>
      <div class="table-block">
        <table class="data-table">
          <thead><tr><th>Contributor</th><th>Contribution</th></tr></thead>
          <tbody>
            <tr><td>Latika</td><td>Final visualization dashboard, embedding overlap analysis, detector/mitigation/RAG plots, error analysis, and failure taxonomy.</td></tr>
            <tr><td>Anushree</td><td>Embeddings, detector metrics, detector scores, mitigation summary, and top-K capped mitigation rows.</td></tr>
            <tr><td>Hardik</td><td>Poisoned KB variants, labels, and change logs used for text-backed taxonomy examples.</td></tr>
            <tr><td>Vatsal</td><td>RAG baseline, undefended, threshold-mitigated, and top-K capped recovery metrics.</td></tr>
            <tr><td>Lance</td><td>Final report integration and narrative synthesis.</td></tr>
          </tbody>
        </table>
      </div>
    </section>
  </main>
</body>
</html>
"""
    REPORT_PATH.write_text(doc, encoding="utf-8")
    return REPORT_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate final visualization tables, figures, ROC curves, notebook, and failure-analysis files.")
    parser.add_argument("--skip-notebook", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    rag, detection, mitigation = read_tables()
    table_paths = write_tables(rag, detection, mitigation)

    figures: List[Path] = []
    figures.extend(copy_embedding_figures())
    figures.append(write_rag_accuracy_figure(rag))
    figures.extend(write_detector_figures(detection))
    mitigation_fig = write_mitigation_figure(mitigation)
    if mitigation_fig is not None:
        figures.append(mitigation_fig)
    figures.extend(write_roc_figures())
    figures.extend(write_lance_png_assets(rag, detection))
    failure_csv, failure_md = write_failure_candidates()
    coverage_report = write_coverage_report(figures, table_paths)
    handoff_notes = write_handoff_notes()
    lance_handoff = write_lance_handoff_package(table_paths)
    html_report = write_html_report(rag, detection, mitigation, figures)
    notebook = None if args.skip_notebook else write_notebook(figures, table_paths)

    print("Wrote tables:")
    for name, path in table_paths.items():
        print(f"  {name}: {path}")
    print("Wrote figures:")
    for path in figures:
        print(f"  {path}")
    print(f"Wrote failure candidates: {failure_csv}")
    print(f"Wrote taxonomy draft: {failure_md}")
    print(f"Wrote coverage report: {coverage_report}")
    print(f"Wrote final integration notes: {handoff_notes}")
    print(f"Wrote final visualization package: {lance_handoff}")
    print(f"Wrote HTML report: {html_report}")
    if notebook is not None:
        print(f"Wrote notebook: {notebook}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
