from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_manifest_path(path_text: str) -> Path:
    return Path(path_text.replace("\\", "/"))


def resolve_image_path(path_text: str, data_root: str | Path) -> Path:
    rel_path = normalize_manifest_path(path_text)
    if rel_path.is_absolute():
        if rel_path.exists():
            return rel_path
        raise FileNotFoundError(f"Image not found: {rel_path}")

    data_root = Path(data_root)
    cwd = Path.cwd()
    search_roots: list[Path] = []
    env_root = os.environ.get("DR_DATA_ROOT")
    if env_root:
        search_roots.append(Path(env_root))
    search_roots.extend([data_root, cwd, cwd.parent])

    candidates: list[Path] = []
    for root in search_roots:
        candidates.extend(
            [
                root / rel_path,
                root / "DR" / rel_path,
                root / "data" / "DR" / rel_path,
                root / "dataset" / "DR" / rel_path,
                root / "\u6570\u636e\u96c6" / "DR" / rel_path,
            ]
        )

    parts = rel_path.parts
    if "DR" in parts:
        dr_index = parts.index("DR")
        dr_tail = Path(*parts[dr_index + 1 :])
        for root in search_roots:
            candidates.extend(
                [
                    root / dr_tail,
                    root / "DR" / dr_tail,
                    root / "data" / "DR" / dr_tail,
                    root / "dataset" / "DR" / dr_tail,
                    root / "\u6570\u636e\u96c6" / "DR" / dr_tail,
                ]
            )

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate

    tried = "\n".join(f"  - {candidate}" for candidate in unique_candidates[:12])
    raise FileNotFoundError(
        f"Image not found for manifest path '{path_text}'. Tried:\n{tried}\n"
        "Set --data-root or DR_DATA_ROOT to the folder that contains BiDR and Messidor-2 Dataset."
    )


class ManifestDataset(Dataset):
    def __init__(self, csv_path: str | Path, data_root: str | Path, transform: Callable | None = None):
        self.csv_path = Path(csv_path)
        self.data_root = Path(data_root)
        self.transform = transform

        with self.csv_path.open(newline="", encoding="utf-8-sig") as f:
            self.rows = list(csv.DictReader(f))

        if not self.rows:
            raise ValueError(f"No rows found in manifest: {self.csv_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image_path = resolve_image_path(row["image_path"], self.data_root)
        label = int(row["label"])

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": label,
            "image_path": str(image_path),
            "dataset": row.get("dataset", ""),
            "source_id": row.get("source_id", image_path.name),
            "diagnosis": row.get("diagnosis", ""),
        }


def build_transforms(image_size: int, train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.88, 1.0), ratio=(0.95, 1.05)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=8),
                transforms.ColorJitter(brightness=0.06, contrast=0.06, saturation=0.04, hue=0.01),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    resize_size = int(round(image_size * 1.14))
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
