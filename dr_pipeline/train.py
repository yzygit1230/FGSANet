from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dr_pipeline.data import ManifestDataset, build_transforms
from dr_pipeline.metrics import classification_metrics, write_predictions
from dr_pipeline.models import create_model, extract_logits, forward_model, parse_model_kwargs


DATASET_MANIFESTS = {
    "BiDR": Path("manifests/dr/by_dataset/BiDR"),
    "Messidor2": Path("manifests/dr/by_dataset/Messidor2"),
}

HISTORY_FIELDS = [
    "epoch",
    "split",
    "loss",
    "auc",
    "macro_auc",
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "macro_precision",
    "macro_recall",
    "macro_f1",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_meta(batch):
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "image_path": [item["image_path"] for item in batch],
        "dataset": [item["dataset"] for item in batch],
        "source_id": [item["source_id"] for item in batch],
        "diagnosis": [item["diagnosis"] for item in batch],
    }


def make_loader(csv_path, data_root, image_size, batch_size, workers, train):
    dataset = ManifestDataset(csv_path, data_root=data_root, transform=build_transforms(image_size, train=train))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_meta,
    )


def num_classes_from_manifest(csv_path: str | Path) -> int:
    declared: set[int] = set()
    labels: set[int] = set()
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("num_classes"):
                declared.add(int(row["num_classes"]))
            labels.add(int(row["label"]))

    if len(declared) > 1:
        raise ValueError(f"Manifest has inconsistent num_classes values: {sorted(declared)}")
    if declared:
        num_classes = declared.pop()
    elif labels:
        num_classes = max(labels) + 1
    else:
        raise ValueError(f"Manifest is empty: {csv_path}")

    if labels and (min(labels) < 0 or max(labels) >= num_classes):
        raise ValueError(f"Labels {sorted(labels)} do not fit num_classes={num_classes}")
    return num_classes


def class_weights_from_manifest(csv_path: str | Path, num_classes: int) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float64)
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            counts[int(row["label"])] += 1

    if np.any(counts == 0):
        missing = [str(idx) for idx, count in enumerate(counts) if count == 0]
        raise ValueError(f"Cannot build class weights because classes are missing from train split: {', '.join(missing)}")

    total = counts.sum()
    weights = total / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        log_pt = F.log_softmax(logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp().clamp(min=1e-6, max=1.0)
        focal_weight = (1.0 - pt).pow(self.gamma)
        return (focal_weight * ce_loss).mean()


def build_criterion(args, weights: torch.Tensor | None) -> nn.Module:
    if args.loss == "ce":
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    if args.loss == "focal":
        return FocalLoss(weight=weights, gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    raise ValueError(f"Unsupported loss: {args.loss}")


def select_binary_threshold(y_true, y_prob, metric_name: str = "balanced_accuracy") -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics: dict[str, float] = {}
    best_score = -float("inf")

    for threshold in np.linspace(0.05, 0.95, 181):
        metrics = classification_metrics(y_true, y_prob, num_classes=2, threshold=float(threshold))
        if metric_name == "youden":
            score = metrics["sensitivity"] + metrics["specificity"] - 1.0
        else:
            score = metrics[metric_name]
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
            best_metrics = metrics

    best_metrics["threshold_metric"] = metric_name
    best_metrics["threshold_score"] = best_score
    return best_threshold, best_metrics


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    amp_enabled: bool,
    num_classes: int,
    pred_path: str | Path | None = None,
    criterion: nn.Module | None = None,
    threshold: float | None = None,
    return_outputs: bool = False,
):
    model.eval()
    y_true, y_prob = [], []
    records = []
    total_loss = 0.0
    criterion = criterion or nn.CrossEntropyLoss()

    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=amp_enabled):
            logits = extract_logits(forward_model(model, images))
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            if num_classes == 2 and threshold is not None:
                preds = (probs[:, 1] >= threshold).long()
            else:
                preds = torch.argmax(probs, dim=1)

        batch_probs = probs.cpu().numpy()
        total_loss += float(loss.item()) * images.size(0)
        y_true.extend(labels.cpu().numpy().tolist())
        y_prob.extend(batch_probs.tolist())

        for i in range(images.size(0)):
            record = {
                "image_path": batch["image_path"][i],
                "dataset": batch["dataset"][i],
                "source_id": batch["source_id"][i],
                "diagnosis": batch["diagnosis"][i],
                "label": int(labels[i].cpu().item()),
                "pred": int(preds[i].cpu().item()),
            }
            for class_idx in range(num_classes):
                record[f"prob_class_{class_idx}"] = float(batch_probs[i, class_idx])
            if num_classes == 2:
                record["prob_dr"] = float(batch_probs[i, 1])
            records.append(record)

    metrics = classification_metrics(y_true, y_prob, num_classes, threshold=threshold)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    if pred_path is not None:
        write_predictions(pred_path, records)
    if return_outputs:
        metrics["_y_true"] = y_true
        metrics["_y_prob"] = y_prob
    return metrics


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, amp_enabled: bool, num_classes: int):
    model.train()
    total_loss = 0.0
    y_true, y_prob = [], []

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=amp_enabled):
            logits = extract_logits(forward_model(model, images, labels=labels))
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item()) * images.size(0)
        probs = torch.softmax(logits.detach(), dim=1)
        y_true.extend(labels.cpu().numpy().tolist())
        y_prob.extend(probs.cpu().numpy().tolist())

    metrics = classification_metrics(y_true, y_prob, num_classes)
    metrics["loss"] = total_loss / max(len(loader.dataset), 1)
    return metrics


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_score, args, model_kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    task_name = "DR_binary_classification" if args.num_classes == 2 else f"DR_{args.num_classes}class_classification"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_score": best_score,
            "model_name": args.model,
            "dataset": args.dataset,
            "task": task_name,
            "model_kwargs": model_kwargs,
            "image_size": args.image_size,
            "num_classes": args.num_classes,
            "loss": args.loss,
            "focal_gamma": args.focal_gamma,
            "label_smoothing": args.label_smoothing,
            "threshold_search": args.threshold_search,
            "threshold_metric": args.threshold_metric,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Train diabetic retinopathy classification models.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_MANIFESTS),
        default=None,
        help="Train one independent DR dataset split. Required unless all CSV paths are given.",
    )
    parser.add_argument("--data-root", default=".", help="Root used to resolve manifest image_path. Use DR_DATA_ROOT for a custom data folder.")
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--val-csv", default=None)
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--output-dir", default="runs/dr_cls")
    parser.add_argument("--model", default="fgsanet", choices=["fgsanet", "ours", "ours_base", "ours_dfga", "ours_sagi"])
    parser.add_argument("--model-kwargs", default="", help="JSON dict forwarded to the model constructor.")
    parser.add_argument("--num-classes", type=int, default=None, help="Override manifest num_classes. By default it is inferred from train.csv.")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--loss", choices=["focal", "ce"], default="focal")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--no-threshold-search", dest="threshold_search", action="store_false")
    parser.add_argument("--threshold-metric", choices=["balanced_accuracy", "f1", "youden"], default="balanced_accuracy")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.dataset is not None:
        manifest_dir = DATASET_MANIFESTS[args.dataset]
        args.train_csv = args.train_csv or str(manifest_dir / "train.csv")
        args.val_csv = args.val_csv or str(manifest_dir / "val.csv")
        args.test_csv = args.test_csv or str(manifest_dir / "test.csv")
    elif not (args.train_csv and args.val_csv and args.test_csv):
        parser.error("Specify --dataset, or provide --train-csv --val-csv --test-csv explicitly.")

    inferred_num_classes = num_classes_from_manifest(args.train_csv)
    if args.num_classes is None:
        args.num_classes = inferred_num_classes
    elif args.num_classes != inferred_num_classes:
        print(f"Warning: --num-classes {args.num_classes} overrides manifest num_classes {inferred_num_classes}")

    set_seed(args.seed)
    output_parts = [Path(args.output_dir)]
    if args.dataset:
        output_parts.append(Path(args.dataset))
    output_parts.append(Path(f"{args.model}_seed{args.seed}"))
    output_dir = Path(*output_parts)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    amp_enabled = (not args.no_amp) and device.type == "cuda"
    model_kwargs = parse_model_kwargs(args.model_kwargs)

    train_loader = make_loader(args.train_csv, args.data_root, args.image_size, args.batch_size, args.workers, train=True)
    val_loader = make_loader(args.val_csv, args.data_root, args.image_size, args.batch_size, args.workers, train=False)
    test_loader = make_loader(args.test_csv, args.data_root, args.image_size, args.batch_size, args.workers, train=False)

    print(
        f"Dataset: {args.dataset or 'custom'} | num_classes: {args.num_classes} "
        f"| image_size: {args.image_size} | loss: {args.loss} "
        f"| focal_gamma: {args.focal_gamma} | label_smoothing: {args.label_smoothing} "
        f"| threshold_search: {args.threshold_search}"
    )
    model = create_model(args.model, num_classes=args.num_classes, image_size=args.image_size, model_kwargs=model_kwargs).to(device)
    weights = None if args.no_class_weights else class_weights_from_manifest(args.train_csv, args.num_classes).to(device)
    criterion = build_criterion(args, weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args) | {"model_kwargs_parsed": model_kwargs}, f, indent=2, ensure_ascii=False)

    history_path = output_dir / "history.csv"
    best_score = -1.0
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"

    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            print(f"Epoch {epoch}/{args.epochs}")
            train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, amp_enabled, args.num_classes)
            val_metrics = evaluate(model, val_loader, device, amp_enabled, args.num_classes, criterion=criterion)
            scheduler.step()

            for split, metrics in (("train", train_metrics), ("val", val_metrics)):
                writer.writerow({"epoch": epoch, "split": split, **{k: metrics.get(k) for k in HISTORY_FIELDS if k not in {"epoch", "split"}}})
            f.flush()

            score = val_metrics["auc"]
            if np.isnan(score):
                score = val_metrics["balanced_accuracy"]
            if score > best_score:
                best_score = float(score)
                save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_score, args, model_kwargs)
            save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_score, args, model_kwargs)

            print(
                f"train loss {train_metrics['loss']:.4f} auc {train_metrics['auc']:.4f} "
                f"| val loss {val_metrics['loss']:.4f} auc {val_metrics['auc']:.4f} "
                f"bal_acc {val_metrics['balanced_accuracy']:.4f} macro_f1 {val_metrics['macro_f1']:.4f}"
            )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    test_threshold = None
    if args.threshold_search and args.num_classes == 2:
        val_for_threshold = evaluate(model, val_loader, device, amp_enabled, args.num_classes, criterion=criterion, return_outputs=True)
        y_true = val_for_threshold.pop("_y_true")
        y_prob = val_for_threshold.pop("_y_prob")
        test_threshold, threshold_metrics = select_binary_threshold(y_true, y_prob, args.threshold_metric)
        with (output_dir / "threshold_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(threshold_metrics, f, indent=2)
        print(f"selected threshold: {test_threshold:.3f} by {args.threshold_metric}")
        print("threshold val metrics:", json.dumps(threshold_metrics, indent=2))

    test_metrics = evaluate(
        model,
        test_loader,
        device,
        amp_enabled,
        args.num_classes,
        pred_path=output_dir / "test_predictions.csv",
        criterion=criterion,
        threshold=test_threshold,
    )
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    print("best checkpoint:", best_path)
    print("test metrics:", json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
