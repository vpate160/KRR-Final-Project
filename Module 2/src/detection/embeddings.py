from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .utils import (
    CLEAN_DOC_IDS,
    CLEAN_EMBEDDINGS,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    M2_EMBEDDINGS,
    load_json,
    load_jsonl,
    save_json,
    set_seed,
    variant_paths,
)

LOGGER = logging.getLogger(__name__)

def _pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"

def encode_texts(
    texts: List[str],
    model_name: str = EMBEDDING_MODEL,
    batch_size: int = 64,
    device: str = "auto",
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    resolved_device = _pick_device(device)
    LOGGER.info("Loading %s on device=%s", model_name, resolved_device)
    model = SentenceTransformer(model_name, device=resolved_device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Unexpected embedding shape {embeddings.shape}; expected (N, {EMBEDDING_DIM})"
        )
    return embeddings.astype(np.float32)

def extract_fields(records: List[Dict]) -> Tuple[List[str], List[str], np.ndarray]:
    doc_ids: List[str] = []
    texts: List[str] = []
    labels: List[int] = []
    for i, rec in enumerate(records):
        doc_id = rec.get("id")
        text = rec.get("text")
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError(f"Record {i} missing valid 'id'")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Record {i} ({doc_id}) missing valid 'text'")
        doc_ids.append(doc_id)
        texts.append(text)
        labels.append(1 if bool(rec.get("is_poisoned", False)) else 0)
    return doc_ids, texts, np.asarray(labels, dtype=np.int8)

def extract_and_save(
    kb_path: Path,
    variant: str,
    batch_size: int = 64,
    device: str = "auto",
    force: bool = False,
) -> Dict[str, Path]:
    paths = variant_paths(variant)

    if not force and all(p.exists() for p in (paths["embeddings"], paths["doc_ids"], paths["labels"])):
        LOGGER.info("Variant '%s' already has all outputs; skipping (pass force=True to overwrite)", variant)
        return paths

    if not kb_path.exists():
        raise FileNotFoundError(f"KB not found for variant '{variant}': {kb_path}")

    LOGGER.info("Loading KB for variant '%s' from %s", variant, kb_path)
    records = load_jsonl(kb_path)
    doc_ids, texts, labels = extract_fields(records)
    LOGGER.info("Loaded %d docs (poisoned=%d)", len(doc_ids), int(labels.sum()))

    set_seed()
    embeddings = encode_texts(texts, batch_size=batch_size, device=device)

    paths["embeddings"].parent.mkdir(parents=True, exist_ok=True)
    np.save(paths["embeddings"], embeddings)
    save_json(paths["doc_ids"], doc_ids)
    np.save(paths["labels"], labels)
    LOGGER.info(
        "Saved embeddings=%s doc_ids=%s labels=%s",
        paths["embeddings"],
        paths["doc_ids"],
        paths["labels"],
    )
    return paths

def ensure_clean_labels() -> Path:
    labels_path = M2_EMBEDDINGS / "clean_labels.npy"
    if labels_path.exists():
        return labels_path
    if not CLEAN_DOC_IDS.exists():
        raise FileNotFoundError(
            f"Clean doc IDs not found at {CLEAN_DOC_IDS}. "
            "Pull Vatsal's latest main — data/embeddings/ should be in the repo."
        )
    doc_ids = load_json(CLEAN_DOC_IDS)
    labels = np.zeros(len(doc_ids), dtype=np.int8)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(labels_path, labels)
    LOGGER.info("Wrote clean labels (all zeros) for %d docs -> %s", len(doc_ids), labels_path)
    return labels_path

def load_variant(variant: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
    paths = variant_paths(variant)
    if variant == "clean":
        ensure_clean_labels()
    missing = [k for k in ("embeddings", "doc_ids", "labels") if not paths[k].exists()]
    if missing:
        raise FileNotFoundError(
            f"Variant '{variant}' missing files: {missing}. "
            "Run extract_and_save() first."
        )
    embeddings = np.load(paths["embeddings"])
    doc_ids = load_json(paths["doc_ids"])
    labels = np.load(paths["labels"])
    if embeddings.shape[0] != len(doc_ids) or embeddings.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Shape mismatch for '{variant}': embeddings={embeddings.shape[0]}, "
            f"doc_ids={len(doc_ids)}, labels={labels.shape[0]}"
        )
    return embeddings, doc_ids, labels

def discover_poisoned_variants(poisoned_dir: Optional[Path] = None) -> List[Tuple[str, Path]]:
    from .utils import M2_POISONED_KB

    root = poisoned_dir or M2_POISONED_KB
    if not root.exists():
        return []
    out: List[Tuple[str, Path]] = []
    for path in sorted(root.glob("poisoned_*.jsonl")):
        variant = path.stem[len("poisoned_"):]
        out.append((variant, path))
    return out
