#!/usr/bin/env python3
"""Task 3.4: Evaluate clean RAG pipeline on Natural Questions.

This script evaluates the existing clean RAG pipeline (Task 3.3) on the first
N examples from Natural Questions, computes EM/F1, and appends metrics to
results/metrics.csv.

Baseline row values are fixed to:
- attack_type=clean
- poison_rate=0.0
- detector=none
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import string
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import chromadb
from datasets import load_dataset
from tqdm import tqdm

from build_rag_pipeline import (
    GEN_BACKEND_HF,
    GEN_BACKEND_OLLAMA,
    OllamaGenerator,
    load_embeddings,
    load_llm,
    list_collection_names,
    resolve_collection_name,
    run_query,
    setup_logging,
)

LOGGER = logging.getLogger("evaluate_rag")

_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = _ARTICLES.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts: Dict[str, int] = {}
    gold_counts: Dict[str, int] = {}

    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1

    common = 0
    for token, count in pred_counts.items():
        common += min(count, gold_counts.get(token, 0))

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def best_em_f1(prediction: str, gold_answers: Sequence[str]) -> Tuple[float, float]:
    if not gold_answers:
        return 0.0, 0.0
    em = max(exact_match(prediction, g) for g in gold_answers)
    f1 = max(token_f1(prediction, g) for g in gold_answers)
    return em, f1


def _clean_text(value: str) -> str:
    return _WHITESPACE.sub(" ", (value or "").strip())


def extract_question_text(example: Dict[str, Any]) -> str:
    q = example.get("question")
    if isinstance(q, dict):
        text = q.get("text", "")
        return _clean_text(str(text))
    return _clean_text(str(q or ""))


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for v in values:
        key = normalize_answer(v)
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def extract_gold_answers(example: Dict[str, Any]) -> List[str]:
    """Extract gold answers from NQ example.

    Priority:
    1) short_answers[*].text[*]
    2) yes_no_answer labels (if present)
    3) long_answer token spans from document tokens
    """

    annotations = example.get("annotations", {})
    if not isinstance(annotations, dict):
        return []

    answers: List[str] = []

    short_answers = annotations.get("short_answers", [])
    if isinstance(short_answers, list):
        for short in short_answers:
            if not isinstance(short, dict):
                continue
            texts = short.get("text", [])
            if not isinstance(texts, list):
                continue
            for t in texts:
                if isinstance(t, str):
                    cleaned = _clean_text(t)
                    if cleaned:
                        answers.append(cleaned)

    yes_no_values = annotations.get("yes_no_answer", [])
    yes_no_map = {1: "yes", 0: "no", 2: "no"}
    if isinstance(yes_no_values, list):
        for yn in yes_no_values:
            if isinstance(yn, int) and yn in yes_no_map:
                answers.append(yes_no_map[yn])
            elif isinstance(yn, str):
                yn_clean = yn.strip().lower()
                if yn_clean in {"yes", "no"}:
                    answers.append(yn_clean)

    # Fallback long answers if short/yes-no are absent.
    if not answers:
        document = example.get("document", {})
        tokens_obj = document.get("tokens", {}) if isinstance(document, dict) else {}
        token_list = tokens_obj.get("token", []) if isinstance(tokens_obj, dict) else []
        is_html_list = tokens_obj.get("is_html", []) if isinstance(tokens_obj, dict) else []

        if isinstance(token_list, list) and isinstance(is_html_list, list) and len(token_list) == len(is_html_list):
            long_answers = annotations.get("long_answer", [])
            if isinstance(long_answers, list):
                for long_a in long_answers:
                    if not isinstance(long_a, dict):
                        continue
                    start = long_a.get("start_token", -1)
                    end = long_a.get("end_token", -1)
                    if not isinstance(start, int) or not isinstance(end, int):
                        continue
                    if start < 0 or end <= start or end > len(token_list):
                        continue

                    span_tokens = [
                        tok
                        for tok, is_html in zip(token_list[start:end], is_html_list[start:end])
                        if not is_html and isinstance(tok, str)
                    ]
                    span = _clean_text(" ".join(span_tokens))
                    if span:
                        answers.append(span)

    return _dedupe_preserve_order(answers)


def append_metrics_row(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp_utc",
        "attack_type",
        "poison_rate",
        "detector",
        "dataset",
        "n_examples",
        "n_answered",
        "n_with_gold",
        "n_failures",
        "exact_match",
        "token_f1",
        "generation_backend",
        "ollama_model",
        "vectorstore_path",
        "collection_requested",
        "collection_resolved",
        "top_k",
    ]

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def evaluate_clean_rag_on_nq(
    *,
    vectorstore_path: str,
    collection_name: str,
    embedding_model: str,
    generation_backend: str,
    model_path: str,
    ollama_model: str,
    ollama_base_url: str,
    top_k: int,
    max_new_tokens: int,
    temperature: float,
    n_examples: int,
    predictions_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Reusable Task 3.4 evaluation entrypoint for clean RAG baseline."""

    ds = load_dataset("natural_questions", split=f"train[:{n_examples}]")

    client = chromadb.PersistentClient(path=vectorstore_path)
    available = list_collection_names(client)
    resolved_collection = resolve_collection_name(collection_name, available)

    LOGGER.info("Available collections: %s", available)
    LOGGER.info("Resolved collection: requested='%s' -> using='%s'", collection_name, resolved_collection)

    embeddings = load_embeddings(embedding_model)
    collection = client.get_collection(name=resolved_collection)

    llm = None
    ollama_generator = None

    if generation_backend == GEN_BACKEND_HF:
        llm = load_llm(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    elif generation_backend == GEN_BACKEND_OLLAMA:
        ollama_generator = OllamaGenerator(
            base_url=ollama_base_url,
            model=ollama_model,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unsupported generation backend: {generation_backend}")

    if predictions_path is not None:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)

    total_em = 0.0
    total_f1 = 0.0
    n_answered = 0
    n_with_gold = 0
    n_failures = 0

    for idx, ex in enumerate(tqdm(ds, desc="Evaluating NQ", unit="q"), start=1):
        question = extract_question_text(ex)
        if not question:
            LOGGER.warning("Skipping empty question at index %d", idx - 1)
            continue

        try:
            pred_answer, _docs = run_query(
                collection=collection,
                embeddings=embeddings,
                generation_backend=generation_backend,
                llm=llm,
                ollama_generator=ollama_generator,
                question=question,
                top_k=top_k,
            )
        except Exception as exc:
            n_failures += 1
            LOGGER.exception("Generation failed at index %d: %s", idx - 1, exc)
            pred_answer = ""

        gold_answers = extract_gold_answers(ex)
        em, f1 = best_em_f1(pred_answer, gold_answers)

        if pred_answer:
            n_answered += 1
        if gold_answers:
            n_with_gold += 1

        total_em += em
        total_f1 += f1

        if predictions_path is not None:
            row = {
                "index": idx - 1,
                "question": question,
                "prediction": pred_answer,
                "gold_answers": gold_answers,
                "em": em,
                "f1": f1,
            }
            with predictions_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_total = len(ds)
    metrics = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "attack_type": "clean",
        "poison_rate": 0.0,
        "detector": "none",
        "dataset": "natural_questions",
        "n_examples": n_total,
        "n_answered": n_answered,
        "n_with_gold": n_with_gold,
        "n_failures": n_failures,
        "exact_match": total_em / n_total if n_total else 0.0,
        "token_f1": total_f1 / n_total if n_total else 0.0,
        "generation_backend": generation_backend,
        "ollama_model": ollama_model if generation_backend == GEN_BACKEND_OLLAMA else "",
        "vectorstore_path": vectorstore_path,
        "collection_requested": collection_name,
        "collection_resolved": resolved_collection,
        "top_k": top_k,
    }
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate clean RAG pipeline on Natural Questions (Task 3.4).")
    parser.add_argument("--vectorstore-path", type=str, default="data/vectorstore/clean")
    parser.add_argument("--collection-name", type=str, default="kb")
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument(
        "--generation-backend",
        type=str,
        default=GEN_BACKEND_OLLAMA,
        choices=[GEN_BACKEND_HF, GEN_BACKEND_OLLAMA],
    )
    parser.add_argument("--model-path", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--ollama-model", type=str, default="llama3.1:8b")
    parser.add_argument("--ollama-base-url", type=str, default="http://localhost:11434")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--n-examples", type=int, default=1000)
    parser.add_argument("--metrics-path", type=str, default="results/metrics.csv")
    parser.add_argument("--predictions-path", type=str, default="results/predictions_clean.jsonl")
    parser.add_argument("--no-predictions", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    predictions_path = None if args.no_predictions else Path(args.predictions_path)
    if predictions_path is not None and predictions_path.exists():
        predictions_path.unlink()

    metrics = evaluate_clean_rag_on_nq(
        vectorstore_path=args.vectorstore_path,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        generation_backend=args.generation_backend,
        model_path=args.model_path,
        ollama_model=args.ollama_model,
        ollama_base_url=args.ollama_base_url,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        n_examples=args.n_examples,
        predictions_path=predictions_path,
    )

    append_metrics_row(Path(args.metrics_path), metrics)

    print("\n=== Evaluation Summary ===")
    print(f"attack_type: {metrics['attack_type']}")
    print(f"poison_rate: {metrics['poison_rate']}")
    print(f"detector: {metrics['detector']}")
    print(f"n_examples: {metrics['n_examples']}")
    print(f"n_answered: {metrics['n_answered']}")
    print(f"n_with_gold: {metrics['n_with_gold']}")
    print(f"n_failures: {metrics['n_failures']}")
    print(f"EM: {metrics['exact_match']:.4f}")
    print(f"token_F1: {metrics['token_f1']:.4f}")
    print(f"metrics_path: {args.metrics_path}")
    if predictions_path is not None:
        print(f"predictions_path: {predictions_path}")


if __name__ == "__main__":
    main()
