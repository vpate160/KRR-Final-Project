#!/usr/bin/env python3
"""Task 3.2: Build a ChromaDB vector store from a clean KB JSONL file.

Collection-name behavior:
- Downstream code should use logical collection name `kb`.
- This script stores vectors in physical collection `kb_store` (explicit/stable).
- For backward compatibility, read/query paths automatically fall back to legacy `kb0`
    if `kb_store` is not present in an existing persisted store.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

LOGGER = logging.getLogger("build_vectorstore")

REQUIRED_FIELDS = ("id", "title", "text", "source", "is_poisoned", "chunk_index")

DEFAULT_LOGICAL_COLLECTION = "kb"
DEFAULT_PHYSICAL_COLLECTION = "kb_store"
LEGACY_PHYSICAL_COLLECTIONS = ("kb0",)


def chroma_physical_collection_name(logical_name: str) -> str:
    name = str(logical_name).strip()
    if name == DEFAULT_LOGICAL_COLLECTION:
        return DEFAULT_PHYSICAL_COLLECTION
    if len(name) >= 3:
        return name

    physical = f"{name}_store"
    LOGGER.warning(
        "Collection '%s' is too short for Chroma; using physical name '%s'.",
        name,
        physical,
    )
    return physical


def resolve_existing_collection_name(client: chromadb.PersistentClient, logical_name: str) -> str:
    preferred = chroma_physical_collection_name(logical_name)
    candidates = [preferred]
    if logical_name == DEFAULT_LOGICAL_COLLECTION:
        candidates.extend(c for c in LEGACY_PHYSICAL_COLLECTIONS if c != preferred)

    for candidate in candidates:
        try:
            client.get_collection(name=candidate)
            if candidate != preferred:
                LOGGER.warning(
                    "Using legacy physical collection '%s' for logical '%s'. "
                    "Recommended rebuild target is '%s'.",
                    candidate,
                    logical_name,
                    preferred,
                )
            return candidate
        except Exception:
            continue

    raise ValueError(
        f"No collection found for logical '{logical_name}'. Tried: {candidates}. "
        "Run the build step to create it."
    )


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def count_lines(file_path: Path) -> int:
    with file_path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def load_kb_jsonl(kb_path: Path) -> List[Dict[str, Any]]:
    if not kb_path.exists():
        raise FileNotFoundError(f"KB file not found: {kb_path}")

    total_lines = count_lines(kb_path)
    LOGGER.info("Reading KB file: %s (%d lines)", kb_path, total_lines)

    records: List[Dict[str, Any]] = []
    with kb_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(tqdm(f, total=total_lines, desc="Loading JSONL", unit="line"), start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_idx}: {exc}") from exc

            missing = [field for field in REQUIRED_FIELDS if field not in obj]
            if missing:
                raise ValueError(f"Line {line_idx} missing required fields: {missing}")

            obj["id"] = str(obj["id"])
            obj["title"] = str(obj["title"])
            obj["text"] = str(obj["text"])
            obj["source"] = str(obj["source"])
            obj["is_poisoned"] = bool(obj["is_poisoned"])
            obj["chunk_index"] = int(obj["chunk_index"])
            records.append(obj)

    if not records:
        raise ValueError("No valid records found in KB JSONL.")

    LOGGER.info("Loaded %d documents from KB.", len(records))
    return records


def build_collection(
    records: Sequence[Dict[str, Any]],
    persist_dir: Path,
    collection_name: str,
    embedding_model_name: str,
    batch_size: int,
    append: bool,
) -> None:
    persist_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading embedding model: %s", embedding_model_name)
    model = SentenceTransformer(embedding_model_name)

    LOGGER.info("Opening Chroma persistent client at: %s", persist_dir)
    client = chromadb.PersistentClient(path=str(persist_dir))

    physical_collection_name = chroma_physical_collection_name(collection_name)

    if not append:
        try:
            client.delete_collection(physical_collection_name)
            LOGGER.info("Deleted existing collection '%s' before rebuild.", physical_collection_name)
        except Exception:
            LOGGER.info("No existing collection '%s' to delete.", physical_collection_name)

    collection = client.get_or_create_collection(
        name=physical_collection_name,
        metadata={"hnsw:space": "cosine", "logical_name": collection_name},
    )

    total = len(records)
    LOGGER.info("Embedding and indexing %d documents in batches of %d.", total, batch_size)

    for start in tqdm(range(0, total, batch_size), desc="Indexing batches", unit="batch"):
        batch = records[start : start + batch_size]

        ids = [rec["id"] for rec in batch]
        documents = [rec["text"] for rec in batch]
        metadatas = [
            {
                "title": rec["title"],
                "source": rec["source"],
                "is_poisoned": rec["is_poisoned"],
                "chunk_index": rec["chunk_index"],
            }
            for rec in batch
        ]

        embeddings = model.encode(
            documents,
            batch_size=min(64, len(documents)),
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
        )

    final_count = collection.count()
    LOGGER.info(
        "Collection logical='%s' physical='%s' now contains %d vectors.",
        collection_name,
        physical_collection_name,
        final_count,
    )


def run_sanity_queries(
    persist_dir: Path,
    collection_name: str,
    embedding_model_name: str,
    top_k: int,
) -> None:
    questions = [
        "Who was Isaac Newton and what was he known for?",
        "What are the causes and effects of the Great Depression?",
        "How is the United Nations organized and what does it do?",
    ]

    model = SentenceTransformer(embedding_model_name)
    client = chromadb.PersistentClient(path=str(persist_dir))
    resolved_name = resolve_existing_collection_name(client, collection_name)
    collection = client.get_collection(name=resolved_name)
    LOGGER.info("Sanity query using collection physical name: %s", resolved_name)

    LOGGER.info("Running retrieval sanity check with %d questions (top_k=%d).", len(questions), top_k)
    for i, q in enumerate(questions, start=1):
        q_embedding = model.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0].tolist()
        result = collection.query(query_embeddings=[q_embedding], n_results=top_k)

        print(f"\nQ{i}: {q}")
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]

        for rank, (doc, meta) in enumerate(zip(docs, metas), start=1):
            title = (meta or {}).get("title", "<unknown>")
            preview = (doc or "").replace("\n", " ")[:180]
            print(f"  {rank}. {title}")
            print(f"     {preview}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ChromaDB vector store from clean KB JSONL.")
    parser.add_argument("--kb-path", type=str, default="data/clean_kb/kb.jsonl")
    parser.add_argument("--persist-dir", type=str, default="data/vectorstore/clean")
    parser.add_argument("--collection-name", type=str, default="kb")
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--append", action="store_true", help="Append to existing collection instead of rebuilding it.")
    parser.add_argument("--sanity-only", action="store_true", help="Skip index build and run only retrieval sanity queries.")
    parser.add_argument("--run-sanity-check", action="store_true")
    parser.add_argument("--sanity-top-k", type=int, default=3)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    kb_path = Path(args.kb_path)
    persist_dir = Path(args.persist_dir)

    if not args.sanity_only:
        records = load_kb_jsonl(kb_path)
        build_collection(
            records=records,
            persist_dir=persist_dir,
            collection_name=args.collection_name,
            embedding_model_name=args.embedding_model,
            batch_size=args.batch_size,
            append=args.append,
        )

    if args.run_sanity_check:
        run_sanity_queries(
            persist_dir=persist_dir,
            collection_name=args.collection_name,
            embedding_model_name=args.embedding_model,
            top_k=args.sanity_top_k,
        )


if __name__ == "__main__":
    main()
