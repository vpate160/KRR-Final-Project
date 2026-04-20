#!/usr/bin/env python3
"""Export clean KB embeddings and aligned document metadata for Module 3 handoff.

Outputs:
- data/embeddings/clean_embeddings.npy
- data/embeddings/clean_doc_ids.json
- data/embeddings/clean_titles.json (optional)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

LOGGER = logging.getLogger("export_clean_embeddings")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export clean KB embeddings and aligned document IDs/titles."
    )
    parser.add_argument("--kb-path", type=str, default="data/clean_kb/kb.jsonl")
    parser.add_argument("--output-dir", type=str, default="data/embeddings")
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for sentence-transformers, e.g. cpu, mps, cuda.",
    )
    parser.add_argument(
        "--normalize-embeddings",
        action="store_true",
        help="L2-normalize embeddings before saving.",
    )
    parser.add_argument(
        "--no-save-titles",
        action="store_true",
        help="Skip saving clean_titles.json.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def load_kb_records(kb_path: Path) -> Tuple[List[str], List[str], List[str]]:
    if not kb_path.exists():
        raise FileNotFoundError(f"KB file not found: {kb_path}")

    doc_ids: List[str] = []
    texts: List[str] = []
    titles: List[str] = []

    with kb_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(tqdm(f, desc="Reading kb.jsonl", unit="line"), start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_num}: {exc}") from exc

            doc_id = rec.get("id")
            text = rec.get("text")
            title = rec.get("title", "")

            if not isinstance(doc_id, str) or not doc_id.strip():
                raise ValueError(f"Missing/invalid 'id' at line {line_num}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Missing/invalid 'text' at line {line_num}")
            if not isinstance(title, str):
                title = str(title)

            doc_ids.append(doc_id)
            texts.append(text)
            titles.append(title)

    if not texts:
        raise ValueError(f"No valid KB records found in {kb_path}")

    LOGGER.info("Loaded %d clean KB documents", len(texts))
    return doc_ids, texts, titles


def export_embeddings(
    kb_path: Path,
    output_dir: Path,
    model_name: str,
    batch_size: int,
    device: str,
    normalize_embeddings: bool,
    save_titles: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    emb_path = output_dir / "clean_embeddings.npy"
    ids_path = output_dir / "clean_doc_ids.json"
    titles_path = output_dir / "clean_titles.json"

    doc_ids, texts, titles = load_kb_records(kb_path)

    LOGGER.info("Loading embedding model: %s (device=%s)", model_name, device)
    model = SentenceTransformer(model_name, device=device)

    LOGGER.info("Encoding %d documents (batch_size=%d)...", len(texts), batch_size)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )

    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape={embeddings.shape}")
    if embeddings.shape[0] != len(doc_ids):
        raise ValueError(
            f"Embedding count mismatch: embeddings={embeddings.shape[0]} ids={len(doc_ids)}"
        )

    np.save(emb_path, embeddings)

    with ids_path.open("w", encoding="utf-8") as f:
        json.dump(doc_ids, f, ensure_ascii=False, indent=2)

    if save_titles:
        with titles_path.open("w", encoding="utf-8") as f:
            json.dump(titles, f, ensure_ascii=False, indent=2)

    LOGGER.info("Saved embeddings: %s", emb_path)
    LOGGER.info("Saved document IDs: %s", ids_path)
    if save_titles:
        LOGGER.info("Saved titles: %s", titles_path)

    LOGGER.info("Export complete: count=%d, dimension=%d", embeddings.shape[0], embeddings.shape[1])


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    export_embeddings(
        kb_path=Path(args.kb_path),
        output_dir=Path(args.output_dir),
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
        normalize_embeddings=args.normalize_embeddings,
        save_titles=not args.no_save_titles,
    )


if __name__ == "__main__":
    main()
