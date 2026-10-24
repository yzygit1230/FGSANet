from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from dr_pipeline.data import ManifestDataset, build_transforms
from dr_pipeline.models import create_model
from dr_pipeline.train import DATASET_MANIFESTS
from dr_pipeline.train import collate_meta, evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained DR classifier checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--threshold", type=float, default=None, help="Binary decision threshold. Defaults to argmax/0.5.")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_name = checkpoint["model_name"]
    dataset_name = checkpoint.get("dataset")
    model_kwargs = checkpoint.get("model_kwargs", {})
    image_size = int(checkpoint.get("image_size", 224))
    num_classes = int(checkpoint.get("num_classes", 2))

    model = create_model(model_name, num_classes=num_classes, image_size=image_size, model_kwargs=model_kwargs).to(device)
    model.load_state_dict(checkpoint["model"])

    csv_path = args.csv
    if csv_path is None and dataset_name in DATASET_MANIFESTS:
        csv_path = str(DATASET_MANIFESTS[dataset_name] / "test.csv")
    if csv_path is None:
        raise ValueError("Pass --csv explicitly, because the checkpoint does not contain a known DR dataset name.")

    dataset = ManifestDataset(csv_path, data_root=args.data_root, transform=build_transforms(image_size, train=False))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_meta,
    )

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent
    threshold = args.threshold
    threshold_file = Path(args.checkpoint).parent / "threshold_metrics.json"
    if threshold is None and num_classes == 2 and threshold_file.exists():
        with threshold_file.open(encoding="utf-8") as f:
            threshold = json.load(f).get("threshold")

    metrics = evaluate(
        model,
        loader,
        device=device,
        amp_enabled=(not args.no_amp) and device.type == "cuda",
        num_classes=num_classes,
        threshold=threshold,
        pred_path=output_dir / "predictions.csv",
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
