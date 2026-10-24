from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    sorted_scores = y_score[order]
    ranks = np.empty_like(sorted_scores, dtype=float)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = (start + 1 + end) / 2.0
        start = end

    full_ranks = np.empty_like(ranks)
    full_ranks[order] = ranks
    sum_pos = float(full_ranks[pos].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def macro_ovr_auc(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> float:
    aucs = []
    for class_idx in range(num_classes):
        class_true = (y_true == class_idx).astype(int)
        auc = binary_auc(class_true, y_prob[:, class_idx])
        if not np.isnan(auc):
            aucs.append(auc)
    return float(np.mean(aucs)) if aucs else float("nan")


def classification_metrics(y_true, y_prob, num_classes: int, threshold: float | None = None) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if num_classes == 2 and threshold is not None:
        y_pred = (y_prob[:, 1] >= threshold).astype(int)
    else:
        y_pred = np.argmax(y_prob, axis=1).astype(int)

    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        if 0 <= true_label < num_classes and 0 <= pred_label < num_classes:
            confusion[true_label, pred_label] += 1

    eps = 1e-12
    support = confusion.sum(axis=1)
    pred_count = confusion.sum(axis=0)
    diag = np.diag(confusion)
    recall = diag / (support + eps)
    precision = diag / (pred_count + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    accuracy = float(diag.sum() / max(confusion.sum(), 1))
    macro_recall = float(np.mean(recall))
    macro_precision = float(np.mean(precision))
    macro_f1 = float(np.mean(f1))
    auc = macro_ovr_auc(y_true, y_prob, num_classes)

    metrics: dict[str, float] = {
        "auc": auc,
        "macro_auc": auc,
        "accuracy": accuracy,
        "balanced_accuracy": macro_recall,
        "macro_recall": macro_recall,
        "sensitivity": macro_recall,
        "macro_precision": macro_precision,
        "precision": macro_precision,
        "macro_f1": macro_f1,
        "f1": macro_f1,
    }
    if threshold is not None:
        metrics["threshold"] = float(threshold)

    for class_idx in range(num_classes):
        metrics[f"class_{class_idx}_support"] = int(support[class_idx])
        metrics[f"class_{class_idx}_precision"] = float(precision[class_idx])
        metrics[f"class_{class_idx}_recall"] = float(recall[class_idx])
        metrics[f"class_{class_idx}_f1"] = float(f1[class_idx])
        for pred_idx in range(num_classes):
            metrics[f"cm_{class_idx}_{pred_idx}"] = int(confusion[class_idx, pred_idx])

    if num_classes == 2:
        tn = int(confusion[0, 0])
        fp = int(confusion[0, 1])
        fn = int(confusion[1, 0])
        tp = int(confusion[1, 1])
        metrics.update(
            {
                "specificity": float(recall[0]),
                "sensitivity": float(recall[1]),
                "precision": float(precision[1]),
                "f1": float(f1[1]),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            }
        )
    else:
        metrics["specificity"] = float("nan")

    return metrics


def write_predictions(path: str | Path, records: list[dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base_fields = ["image_path", "dataset", "source_id", "diagnosis", "label", "pred"]
    extra_fields = sorted({key for record in records for key in record if key not in base_fields})
    fieldnames = base_fields + extra_fields
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
