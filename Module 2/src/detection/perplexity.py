from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .anomaly_detector import (
    DetectorResult,
    _attack_type_from_variant,
    _poison_rate_from_variant,
    log_result,
    save_scores,
)
from .embeddings import load_variant
from .utils import SEED, load_jsonl, set_seed, variant_paths

LOGGER = logging.getLogger(__name__)

PERPLEXITY = "perplexity"

def _load_gpt2(model_name: str = "gpt2", device: Optional[str] = None):
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    LOGGER.info("Loading %s on device=%s", model_name, device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()
    return tokenizer, model, device

def _doc_nll_sliding(
    text: str,
    tokenizer,
    model,
    device: str,
    max_length: int = 1024,
    stride: int = 512,
) -> float:
    import torch

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.size(1)
    if seq_len == 0:
        return float("nan")

    nlls: List[torch.Tensor] = []
    token_counts: List[int] = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        ids = input_ids[:, begin_loc:end_loc]
        target_ids = ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * max(trg_len - 1, 1)
        nlls.append(neg_log_likelihood)
        token_counts.append(max(trg_len - 1, 1))
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    total_nll = torch.stack(nlls).sum().item()
    total_tokens = sum(token_counts)
    return total_nll / max(total_tokens, 1)

def compute_perplexities(
    texts: List[str],
    model_name: str = "gpt2",
    device: Optional[str] = None,
) -> np.ndarray:
    tokenizer, model, device = _load_gpt2(model_name, device)
    out = np.empty(len(texts), dtype=np.float32)
    for i, t in enumerate(texts):
        avg_nll = _doc_nll_sliding(t, tokenizer, model, device)
        out[i] = float(math.exp(avg_nll)) if math.isfinite(avg_nll) else float("inf")
        if (i + 1) % 200 == 0:
            LOGGER.info("scored %d/%d (latest ppl=%.2f)", i + 1, len(texts), out[i])
    return out

def _threshold_top_k(scores: np.ndarray, k_frac: float) -> Tuple[float, np.ndarray]:
    if not 0.0 < k_frac < 1.0:
        raise ValueError(f"k_frac must be in (0, 1); got {k_frac}")
    cutoff = float(np.quantile(scores, 1.0 - k_frac))
    preds = (scores >= cutoff).astype(int)
    return cutoff, preds

def evaluate_variant(
    variant: str,
    model_name: str = "gpt2",
    k_frac: Optional[float] = None,
    device: Optional[str] = None,
) -> DetectorResult:
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    set_seed()
    paths = variant_paths(variant)
    if not paths["kb"].exists():
        raise FileNotFoundError(f"KB file not found for variant '{variant}': {paths['kb']}")

    LOGGER.info("Loading KB for variant '%s' from %s", variant, paths["kb"])
    records = load_jsonl(paths["kb"])
    texts = [r["text"] for r in records]

    _, _, labels = load_variant(variant)
    if len(labels) != len(texts):
        raise ValueError(
            f"Label/KB length mismatch for '{variant}': labels={len(labels)} texts={len(texts)}"
        )

    if k_frac is None:
        rate = _poison_rate_from_variant(variant)
        if rate <= 0.0:
            rate = float(labels.mean())
        k_frac = max(rate, 0.01)
    LOGGER.info("Using top-K flagging with K=%.3f", k_frac)

    scores = compute_perplexities(texts, model_name=model_name, device=device)
    cutoff, preds = _threshold_top_k(scores, k_frac)

    if labels.sum() == 0 or labels.sum() == len(labels):
        auc = float("nan")
    else:
        auc = float(roc_auc_score(labels, scores))

    result = DetectorResult(
        detector=PERPLEXITY,
        variant=variant,
        scores=scores.astype(np.float32),
        preds=preds.astype(np.int8),
        labels=labels.astype(np.int8),
        precision=float(precision_score(labels, preds, zero_division=0)),
        recall=float(recall_score(labels, preds, zero_division=0)),
        f1=float(f1_score(labels, preds, zero_division=0)),
        roc_auc=auc,
        threshold_desc=f"top{int(round(k_frac * 100))}pct@ppl>={cutoff:.2f}",
    )
    LOGGER.info(
        "[perplexity @ %s] P=%.3f R=%.3f F1=%.3f AUC=%.3f cutoff_ppl=%.2f",
        variant,
        result.precision,
        result.recall,
        result.f1,
        result.roc_auc,
        cutoff,
    )
    save_scores(result)
    log_result(result, notes=f"model={model_name}")
    return result

def evaluate_variants(
    variants: List[str],
    model_name: str = "gpt2",
    k_frac: Optional[float] = None,
    device: Optional[str] = None,
) -> List[DetectorResult]:
    results: List[DetectorResult] = []
    for v in variants:
        if v == "clean":
            continue
        results.append(evaluate_variant(v, model_name=model_name, k_frac=k_frac, device=device))
    return results
