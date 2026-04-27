from __future__ import annotations

import csv
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

MODULE2_ROOT: Path = Path(__file__).resolve().parents[2]
REPO_ROOT: Path = MODULE2_ROOT.parent

CLEAN_KB: Path = REPO_ROOT / "data" / "clean_kb" / "kb.jsonl"
CLEAN_EMBEDDINGS: Path = REPO_ROOT / "data" / "embeddings" / "clean_embeddings.npy"
CLEAN_DOC_IDS: Path = REPO_ROOT / "data" / "embeddings" / "clean_doc_ids.json"
CLEAN_TITLES: Path = REPO_ROOT / "data" / "embeddings" / "clean_titles.json"
CLEAN_VECTORSTORE: Path = REPO_ROOT / "data" / "vectorstore" / "clean"
MODULE3_SCRIPTS: Path = REPO_ROOT / "scripts"
ROOT_METRICS_CSV: Path = REPO_ROOT / "results" / "metrics.csv"

M2_EMBEDDINGS: Path = MODULE2_ROOT / "data" / "embeddings"
M2_POISONED_KB: Path = MODULE2_ROOT / "data" / "poisoned_kb"
M2_FILTERED_KB: Path = MODULE2_ROOT / "data" / "filtered_kb"
M2_VECTORSTORES: Path = MODULE2_ROOT / "data" / "vectorstores"
M2_MODELS: Path = MODULE2_ROOT / "models"
M2_SCORES: Path = MODULE2_ROOT / "results" / "scores"
M2_DETECTION_CSV: Path = MODULE2_ROOT / "results" / "detection_metrics.csv"

HARDIK_POISONED_KB: Path = REPO_ROOT / "data" / "poisoned_kb"

POISONED_KB_SEARCH_DIRS: List[Path] = [HARDIK_POISONED_KB, M2_POISONED_KB]

EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM: int = 768
SEED: int = 42

DETECTION_CSV_FIELDS: List[str] = [
    "timestamp_utc",
    "experiment_id",
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

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def save_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def append_detection_row(row: Dict[str, Any], csv_path: Path = M2_DETECTION_CSV) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    full_row = {field: row.get(field, "") for field in DETECTION_CSV_FIELDS}
    full_row.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
    full_row["timestamp_utc"] = full_row["timestamp_utc"] or datetime.now(timezone.utc).isoformat()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DETECTION_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(full_row)

def resolve_kb_path(variant: str) -> Path:
    candidates = []
    for root in POISONED_KB_SEARCH_DIRS:
        candidates.append(root / f"{variant}.jsonl")
        candidates.append(root / f"poisoned_{variant}.jsonl")
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def resolve_external_labels_npy(variant: str) -> Optional[Path]:
    for root in POISONED_KB_SEARCH_DIRS:
        candidate = root / f"{variant}_labels.npy"
        if candidate.exists():
            return candidate
    return None


def variant_paths(variant: str) -> Dict[str, Path]:
    if variant == "clean":
        return {
            "embeddings": CLEAN_EMBEDDINGS,
            "doc_ids": CLEAN_DOC_IDS,
            "labels": M2_EMBEDDINGS / "clean_labels.npy",
            "kb": CLEAN_KB,
        }
    return {
        "embeddings": M2_EMBEDDINGS / f"{variant}_embeddings.npy",
        "doc_ids": M2_EMBEDDINGS / f"{variant}_doc_ids.json",
        "labels": M2_EMBEDDINGS / f"{variant}_labels.npy",
        "kb": resolve_kb_path(variant),
    }
