from __future__ import annotations

import logging
from dataclasses import dataclass
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
from .utils import EMBEDDING_DIM, M2_MODELS, SEED, set_seed

LOGGER = logging.getLogger(__name__)

NEURAL = "neural_classifier"

@dataclass
class TrainingHistory:
    train_loss: List[float]
    val_loss: List[float]
    best_epoch: int
    best_val_loss: float
    threshold: float

def _make_model(input_dim: int = EMBEDDING_DIM) -> "torch.nn.Module":
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )

def _stratified_split(
    labels: np.ndarray,
    train_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=seed,
        stratify=labels,
    )
    return train_idx, test_idx

def _pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def train_classifier(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int = SEED,
    train_ratio: float = 0.70,
    val_ratio_of_train: float = 0.15,
    max_epochs: int = 30,
    patience: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: Optional[str] = None,
) -> Tuple["torch.nn.Module", Dict[str, np.ndarray], TrainingHistory]:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    set_seed(seed)
    resolved_device = device or _pick_device()

    if labels.sum() == 0:
        raise ValueError("Cannot train classifier: labels contain no positive examples.")
    if labels.sum() == len(labels):
        raise ValueError("Cannot train classifier: labels contain no negative examples.")

    train_idx_full, test_idx = _stratified_split(labels, train_ratio, seed)
    sub_labels = labels[train_idx_full]
    inner_train_ratio = 1.0 - val_ratio_of_train
    inner_train_rel, val_rel = _stratified_split(sub_labels, inner_train_ratio, seed + 1)
    train_idx = train_idx_full[inner_train_rel]
    val_idx = train_idx_full[val_rel]

    LOGGER.info(
        "Split sizes: train=%d val=%d test=%d (pos rates: %.3f / %.3f / %.3f)",
        len(train_idx),
        len(val_idx),
        len(test_idx),
        float(labels[train_idx].mean()),
        float(labels[val_idx].mean()),
        float(labels[test_idx].mean()),
    )

    def _to_tensor(arr: np.ndarray, dtype) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(arr)).to(dtype)

    x_train = _to_tensor(embeddings[train_idx], torch.float32)
    y_train = _to_tensor(labels[train_idx], torch.float32).view(-1, 1)
    x_val = _to_tensor(embeddings[val_idx], torch.float32)
    y_val = _to_tensor(labels[val_idx], torch.float32).view(-1, 1)

    n_pos = float(y_train.sum().item())
    n_neg = float(y_train.numel() - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32)
    LOGGER.info(
        "Class balance: n_pos=%d n_neg=%d -> pos_weight=%.3f", int(n_pos), int(n_neg), pos_weight.item()
    )

    model = _make_model().to(resolved_device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(resolved_device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    x_val = x_val.to(resolved_device)
    y_val = y_val.to(resolved_device)

    best_state = None
    best_val = float("inf")
    best_epoch = -1
    train_losses: List[float] = []
    val_losses: List[float] = []
    epochs_since_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb = xb.to(resolved_device)
            yb = yb.to(resolved_device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(loader.dataset)

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_loss = criterion(val_logits, y_val).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        LOGGER.info("epoch=%d train_loss=%.4f val_loss=%.4f", epoch, train_loss, val_loss)

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                LOGGER.info("Early stopping at epoch %d (best=%d)", epoch, best_epoch)
                break

    if best_state is None:
        raise RuntimeError("Training failed: no best state captured.")
    model.load_state_dict(best_state)

    threshold = _pick_threshold_on_val(model, x_val, y_val, resolved_device)
    history = TrainingHistory(
        train_loss=train_losses,
        val_loss=val_losses,
        best_epoch=best_epoch,
        best_val_loss=best_val,
        threshold=threshold,
    )
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    return model, splits, history

def _pick_threshold_on_val(
    model: "torch.nn.Module",
    x_val: "torch.Tensor",
    y_val: "torch.Tensor",
    device: str,
) -> float:
    import torch
    from sklearn.metrics import f1_score

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x_val)).cpu().numpy().ravel()
    labels = y_val.cpu().numpy().ravel().astype(int)

    best_f1 = -1.0
    best_t = 0.5
    for t in np.linspace(0.05, 0.95, 19):
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    LOGGER.info("Best val threshold=%.2f (val F1=%.3f)", best_t, best_f1)
    return best_t

def score_test(
    model: "torch.nn.Module",
    embeddings: np.ndarray,
    labels: np.ndarray,
    test_idx: np.ndarray,
    threshold: float,
    variant: str,
) -> DetectorResult:
    import torch
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    device = next(model.parameters()).device
    x_test = torch.from_numpy(np.ascontiguousarray(embeddings[test_idx])).to(torch.float32).to(device)
    y_test = labels[test_idx].astype(int)

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x_test)).cpu().numpy().ravel()
    preds = (probs >= threshold).astype(int)

    if y_test.sum() == 0 or y_test.sum() == len(y_test):
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_test, probs))

    return DetectorResult(
        detector=NEURAL,
        variant=variant,
        scores=probs.astype(np.float32),
        preds=preds.astype(np.int8),
        labels=y_test.astype(np.int8),
        precision=float(precision_score(y_test, preds, zero_division=0)),
        recall=float(recall_score(y_test, preds, zero_division=0)),
        f1=float(f1_score(y_test, preds, zero_division=0)),
        roc_auc=auc,
        threshold_desc=f"{threshold:.2f}",
    )

def _save_model(model: "torch.nn.Module", variant: str, history: TrainingHistory) -> Path:
    import torch

    M2_MODELS.mkdir(parents=True, exist_ok=True)
    out = M2_MODELS / f"{variant}_mlp.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "threshold": history.threshold,
            "best_epoch": history.best_epoch,
            "best_val_loss": history.best_val_loss,
        },
        out,
    )
    LOGGER.info("Saved MLP checkpoint -> %s", out)
    return out

def train_and_evaluate_variant(
    variant: str,
    *,
    max_epochs: int = 30,
    patience: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> DetectorResult:
    if variant == "clean":
        raise ValueError("Cannot train supervised classifier on the 'clean' variant (no positives).")

    embeddings, _, labels = load_variant(variant)
    model, splits, history = train_classifier(
        embeddings=embeddings,
        labels=labels,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        lr=lr,
    )
    _save_model(model, variant, history)
    result = score_test(
        model=model,
        embeddings=embeddings,
        labels=labels,
        test_idx=splits["test"],
        threshold=history.threshold,
        variant=variant,
    )
    LOGGER.info(
        "[neural @ %s] P=%.3f R=%.3f F1=%.3f AUC=%.3f threshold=%.2f",
        variant,
        result.precision,
        result.recall,
        result.f1,
        result.roc_auc,
        history.threshold,
    )
    save_scores(result)
    log_result(
        result,
        notes=f"best_epoch={history.best_epoch} best_val_loss={history.best_val_loss:.4f}",
    )
    return result

def evaluate_variants(variants: List[str], **kwargs) -> List[DetectorResult]:
    results: List[DetectorResult] = []
    for v in variants:
        if v == "clean":
            LOGGER.info("Skipping 'clean' variant for supervised training.")
            continue
        results.append(train_and_evaluate_variant(v, **kwargs))
    return results
