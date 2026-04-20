#!/usr/bin/env python3
"""Build a clean Wikipedia KB aligned to Natural Questions (Task 3.1)."""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import unquote, urlparse

import wikipediaapi
from datasets import load_dataset
from tqdm import tqdm


LOGGER = logging.getLogger("build_kb")

CANONICAL_TITLE_PATH_HINTS = (
    "document.title",
    "document_title",
    "document.url",
    "document_url",
    "wikipedia_url",
    "wiki_url",
)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def normalize_title(title: str) -> str:
    title = unquote(title or "")
    title = title.replace("_", " ").strip()
    title = re.sub(r"\s+", " ", title)
    return title


def is_obviously_noisy_title(title: str) -> bool:
    t = normalize_title(title)
    tl = t.lower()
    return tl.startswith("list of") or tl.startswith("category:") or "(disambiguation)" in tl


def title_from_wikipedia_url(value: str) -> Optional[str]:
    if not isinstance(value, str) or "wikipedia.org" not in value:
        return None
    try:
        parsed = urlparse(value)
        if "wikipedia.org" not in parsed.netloc:
            return None
        if "/wiki/" not in parsed.path:
            return None
        title = parsed.path.split("/wiki/", 1)[1]
        title = title.split("#", 1)[0]
        title = normalize_title(title)
        return title or None
    except Exception:
        return None


def clean_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text
    cleaned = re.sub(r"\[[0-9]+\]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip()
    return cleaned


def chunk_text(text: str, chunk_words: int = 300, min_chunk_words: int = 80) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    for start in range(0, len(words), chunk_words):
        part = words[start : start + chunk_words]
        if len(part) < min_chunk_words and start != 0:
            continue
        if len(part) < min_chunk_words and len(words) >= min_chunk_words:
            continue
        if len(part) < min_chunk_words and len(words) < min_chunk_words:
            return []
        chunks.append(" ".join(part))
    return chunks


def recursively_collect_title_candidates(obj: Any, path: str = "") -> List[Tuple[str, str]]:
    candidates: List[Tuple[str, str]] = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            key_l = key.lower()
            new_path = f"{path}.{key}" if path else key

            if isinstance(value, str):
                if "title" in key_l:
                    title = normalize_title(value)
                    if title:
                        candidates.append((new_path, title))
                if any(t in key_l for t in ("url", "uri", "link", "page")):
                    parsed = title_from_wikipedia_url(value)
                    if parsed:
                        candidates.append((new_path, parsed))

            candidates.extend(recursively_collect_title_candidates(value, new_path))

    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            new_path = f"{path}[{idx}]" if path else f"[{idx}]"
            candidates.extend(recursively_collect_title_candidates(item, new_path))

    elif isinstance(obj, str):
        parsed = title_from_wikipedia_url(obj)
        if parsed:
            candidates.append((path or "<root>", parsed))

    return candidates


def inspect_nq_schema(dataset: Any, sample_size: int = 5, path_probe_size: int = 200) -> Counter:
    LOGGER.info("Dataset size loaded: %d", len(dataset))
    LOGGER.info("Dataset columns: %s", dataset.column_names)

    try:
        feature_keys = list(dataset.features.keys())
        LOGGER.info("Top-level feature keys: %s", feature_keys)
    except Exception:
        LOGGER.info("Could not read dataset.features keys cleanly; proceeding with sample inspection.")

    path_counter: Counter = Counter()
    probe_n = min(path_probe_size, len(dataset))
    for i in range(probe_n):
        candidates = recursively_collect_title_candidates(dataset[i])
        for path, _title in candidates:
            path_counter[path] += 1

    if path_counter:
        LOGGER.info("Likely title-containing paths (top 10): %s", path_counter.most_common(10))
    else:
        LOGGER.warning("No obvious title paths discovered from samples.")

    for i in range(min(sample_size, len(dataset))):
        sample = dataset[i]
        candidates = recursively_collect_title_candidates(sample)
        shown = [f"{p} -> {t}" for p, t in candidates[:5]]
        LOGGER.info("Sample %d title candidates: %s", i, shown if shown else "None")

    return path_counter


def load_nq_examples(n_examples: int) -> Any:
    split = f"train[:{n_examples}]"
    attempts = [
        ("natural_questions", None),
        ("natural_questions", "default"),
        ("google-research-datasets/natural_questions", None),
    ]

    errors: List[str] = []
    for dataset_name, config_name in attempts:
        try:
            LOGGER.info(
                "Trying to load dataset: %s%s, split=%s",
                dataset_name,
                f" (config={config_name})" if config_name else "",
                split,
            )
            if config_name:
                ds = load_dataset(dataset_name, config_name, split=split)
            else:
                ds = load_dataset(dataset_name, split=split)
            LOGGER.info("Successfully loaded %d examples from %s", len(ds), dataset_name)
            return ds
        except Exception as exc:
            msg = f"{dataset_name} config={config_name}: {exc}"
            errors.append(msg)
            LOGGER.warning("Failed loading %s", msg)

    joined = "\n".join(errors)
    raise RuntimeError(
        "Could not load Natural Questions from known identifiers. "
        "Please verify internet access and datasets package version.\n"
        f"Attempts:\n{joined}"
    )


def extract_seed_titles(dataset: Any, max_examples: int) -> List[str]:
    path_counter = inspect_nq_schema(dataset)

    def candidate_priority(path: str) -> Tuple[int, int, int, str]:
        path_l = path.lower()
        canonical_score = 0 if any(hint in path_l for hint in CANONICAL_TITLE_PATH_HINTS) else 1
        title_like_score = 0 if "title" in path_l else 1
        frequent_path_score = -path_counter.get(path, 0)
        return (canonical_score, title_like_score, frequent_path_score, path_l)

    titles: List[str] = []
    seen: Set[str] = set()

    n = min(max_examples, len(dataset))
    iterator = tqdm(range(n), desc="Extracting seed titles", unit="example")

    for i in iterator:
        example = dataset[i]
        candidates = recursively_collect_title_candidates(example)

        candidates.sort(key=lambda pt: candidate_priority(pt[0]))

        for path, raw_title in candidates:
            _ = path
            title = normalize_title(raw_title)
            if not title:
                continue
            if len(title) > 250:
                continue
            if title.lower().startswith("http"):
                continue
            if is_obviously_noisy_title(title):
                continue
            key = title.casefold()
            if key in seen:
                continue
            seen.add(key)
            titles.append(title)
            break

    LOGGER.info("Extracted %d unique seed titles from first %d examples", len(titles), n)
    return titles


def create_wiki_client(user_agent: str) -> wikipediaapi.Wikipedia:
    return wikipediaapi.Wikipedia(
        language="en",
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        user_agent=user_agent,
    )


def fetch_wikipedia_article(
    wiki: wikipediaapi.Wikipedia,
    title: str,
    request_delay: float,
    max_retries: int = 3,
) -> Optional[Tuple[str, str, List[str]]]:
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            page = wiki.page(title)
            if not page.exists():
                time.sleep(request_delay)
                return None

            resolved_title = normalize_title(page.title)
            text = clean_text(page.text)
            if not text:
                time.sleep(request_delay)
                return None

            linked = [normalize_title(t) for t in page.links.keys() if normalize_title(t)]
            time.sleep(request_delay)
            return resolved_title, text, linked
        except Exception as exc:
            last_error = exc
            backoff = request_delay * attempt
            LOGGER.warning("Fetch failed for '%s' (attempt %d/%d): %s", title, attempt, max_retries, exc)
            time.sleep(backoff)

    if last_error:
        LOGGER.error("Giving up on '%s': %s", title, last_error)
    return None


def make_document(doc_number: int, title: str, chunk_text_value: str, chunk_index: int) -> Dict[str, Any]:
    return {
        "id": f"wiki_{doc_number:05d}_chunk_{chunk_index:03d}",
        "title": title,
        "text": chunk_text_value,
        "source": "wikipedia",
        "is_poisoned": False,
        "chunk_index": chunk_index,
    }


def add_article_chunks(
    documents: List[Dict[str, Any]],
    dedupe_keys: Set[Tuple[str, int]],
    title: str,
    article_text: str,
    chunk_words: int,
    min_chunk_words: int,
    target_docs: int,
    max_chunks_per_title: int,
) -> int:
    chunks = chunk_text(article_text, chunk_words=chunk_words, min_chunk_words=min_chunk_words)
    if max_chunks_per_title > 0:
        chunks = chunks[:max_chunks_per_title]
    added = 0

    for chunk_idx, chunk in enumerate(chunks):
        key = (title.casefold(), chunk_idx)
        if key in dedupe_keys:
            continue
        if len(documents) >= target_docs:
            break

        dedupe_keys.add(key)
        doc_num = len(documents) + 1
        documents.append(make_document(doc_num, title, chunk, chunk_idx))
        added += 1

    return added


def deduplicate_documents(documents: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[Tuple[str, int]] = set()
    deduped: List[Dict[str, Any]] = []

    for doc in documents:
        key = (str(doc["title"]).casefold(), int(doc["chunk_index"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)

    for idx, doc in enumerate(deduped, start=1):
        doc["id"] = f"wiki_{idx:05d}_chunk_{int(doc['chunk_index']):03d}"

    return deduped


def write_jsonl(records: Iterable[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def expand_with_one_hop_links(
    wiki: wikipediaapi.Wikipedia,
    seed_titles: Sequence[str],
    already_fetched: Set[str],
    request_delay: float,
    max_seed_pages_for_links: int = 400,
) -> List[str]:
    link_titles: List[str] = []
    seen_links: Set[str] = set()

    subset = seed_titles[:max_seed_pages_for_links]
    for title in tqdm(subset, desc="Collecting one-hop links", unit="seed"):
        result = fetch_wikipedia_article(wiki, title, request_delay=request_delay)
        if not result:
            continue
        resolved_title, _text, links = result
        already_fetched.add(resolved_title.casefold())

        for link in links:
            key = link.casefold()
            if is_obviously_noisy_title(link):
                continue
            if key in seen_links or key in already_fetched:
                continue
            seen_links.add(key)
            link_titles.append(link)

    LOGGER.info("Collected %d unique one-hop linked titles", len(link_titles))
    return link_titles


def build_kb(args: argparse.Namespace) -> None:
    dataset = load_nq_examples(args.nq_examples)
    seed_titles = extract_seed_titles(dataset, max_examples=args.nq_examples)

    if args.max_seed_titles > 0:
        seed_titles = seed_titles[: args.max_seed_titles]
        LOGGER.info("Trimmed seed titles to max_seed_titles=%d", args.max_seed_titles)

    wiki = create_wiki_client(user_agent=args.user_agent)

    documents: List[Dict[str, Any]] = []
    dedupe_keys: Set[Tuple[str, int]] = set()
    fetched_titles: Set[str] = set()

    seed_progress = tqdm(seed_titles, desc="Fetching seed articles", unit="article")
    for title in seed_progress:
        if is_obviously_noisy_title(title):
            continue
        if len(documents) >= args.target_docs:
            break
        if title.casefold() in fetched_titles:
            continue

        result = fetch_wikipedia_article(wiki, title, request_delay=args.request_delay)
        if not result:
            continue

        resolved_title, text, _links = result
        fetched_titles.add(resolved_title.casefold())

        added = add_article_chunks(
            documents,
            dedupe_keys,
            resolved_title,
            text,
            chunk_words=args.chunk_words,
            min_chunk_words=args.min_chunk_words,
            target_docs=args.target_docs,
            max_chunks_per_title=args.max_chunks_per_title,
        )
        seed_progress.set_postfix({"docs": len(documents), "last_added": added})

    if len(documents) < args.target_docs:
        LOGGER.info(
            "Seed set produced %d docs (< %d). Expanding via one-hop linked articles.",
            len(documents),
            args.target_docs,
        )
        linked_titles = expand_with_one_hop_links(
            wiki,
            seed_titles,
            already_fetched=fetched_titles,
            request_delay=args.request_delay,
            max_seed_pages_for_links=args.max_seed_pages_for_links,
        )

        link_progress = tqdm(linked_titles, desc="Fetching linked articles", unit="article")
        for title in link_progress:
            if is_obviously_noisy_title(title):
                continue
            if len(documents) >= args.target_docs:
                break
            if title.casefold() in fetched_titles:
                continue

            result = fetch_wikipedia_article(wiki, title, request_delay=args.request_delay)
            if not result:
                continue

            resolved_title, text, _links = result
            fetched_titles.add(resolved_title.casefold())

            added = add_article_chunks(
                documents,
                dedupe_keys,
                resolved_title,
                text,
                chunk_words=args.chunk_words,
                min_chunk_words=args.min_chunk_words,
                target_docs=args.target_docs,
                max_chunks_per_title=args.max_chunks_per_title,
            )
            link_progress.set_postfix({"docs": len(documents), "last_added": added})

    documents = deduplicate_documents(documents)
    if len(documents) > args.target_docs:
        documents = documents[: args.target_docs]
        documents = deduplicate_documents(documents)

    output_path = Path(args.output_path)
    write_jsonl(documents, output_path)

    LOGGER.info("Wrote %d documents to %s", len(documents), output_path)
    if len(documents) < args.target_docs:
        LOGGER.warning(
            "Output has fewer than target docs (%d < %d). Consider lowering min_chunk_words, "
            "increasing max_seed_titles, or re-running with better network/API access.",
            len(documents),
            args.target_docs,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean Wikipedia KB for NQ-aligned RAG.")
    parser.add_argument("--output-path", type=str, default="data/clean_kb/kb.jsonl")
    parser.add_argument("--target-docs", type=int, default=5000)
    parser.add_argument("--nq-examples", type=int, default=1000)
    parser.add_argument("--chunk-words", type=int, default=300)
    parser.add_argument("--min-chunk-words", type=int, default=80)
    parser.add_argument("--max-chunks-per-title", type=int, default=25)
    parser.add_argument("--max-seed-titles", type=int, default=0, help="0 means no explicit cap")
    parser.add_argument("--max-seed-pages-for-links", type=int, default=400)
    parser.add_argument("--request-delay", type=float, default=0.2)
    parser.add_argument(
        "--user-agent",
        type=str,
        default="KRR-Project-Task3.1-KBBuilder/1.0 (student project; contact: local)",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    build_kb(args)


if __name__ == "__main__":
    main()
