from __future__ import annotations

import argparse
import csv
import os
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "manifests" / "dr" / "by_dataset"
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42


def find_dr_root(dr_root: str | Path | None = None) -> Path:
    if dr_root:
        candidate = Path(dr_root).expanduser()
        if (candidate / "BiDR").exists() and (candidate / "Messidor-2 Dataset").exists():
            return candidate
        raise FileNotFoundError(f"--dr-root must contain BiDR and Messidor-2 Dataset: {candidate}")

    dataset_dir = "\u6570\u636e\u96c6"
    env_root = os.environ.get("DR_DATA_ROOT")
    candidates = [
        Path(env_root).expanduser() if env_root else None,
        ROOT / dataset_dir / "DR",
        ROOT.parent / dataset_dir / "DR",
        ROOT / "dataset" / "DR",
        ROOT.parent / "dataset" / "DR",
        ROOT / "data" / "DR",
        ROOT.parent / "data" / "DR",
        ROOT / "DR",
        ROOT.parent / "DR",
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "BiDR").exists() and (candidate / "Messidor-2 Dataset").exists():
            return candidate
    tried = "\n".join(f"  - {candidate}" for candidate in candidates if candidate is not None)
    raise FileNotFoundError(f"Could not find DR data root. Tried:\n{tried}")


DR_ROOT = Path()


def label_name_bidr(label: int) -> str:
    return "DR" if label == 1 else "No_DR"


def label_name_messidor(label: int) -> str:
    return "DR" if label == 1 else "No_DR"


def split_counts(n: int) -> dict[str, int]:
    raw = {split: n * ratio for split, ratio in RATIOS.items()}
    counts = {split: int(value) for split, value in raw.items()}
    missing = n - sum(counts.values())
    by_remainder = sorted(RATIOS, key=lambda split: (raw[split] - counts[split], RATIOS[split]), reverse=True)
    for split in by_remainder[:missing]:
        counts[split] += 1
    return counts


def read_bidr() -> list[dict[str, str]]:
    root = DR_ROOT / "BiDR" / "retino"
    rows: list[dict[str, str]] = []
    split_map = {"train": "train", "valid": "val", "test": "test"}
    class_map = {"No_DR": 0, "DR": 1}

    for source_split, split in split_map.items():
        for class_name, label in class_map.items():
            class_dir = root / source_split / class_name
            for image_path in sorted(class_dir.glob("*")):
                if not image_path.is_file():
                    continue
                rows.append(
                    {
                        "split": split,
                        "dataset": "BiDR",
                        "task": "binary",
                        "num_classes": "2",
                        "image_variant": "retino",
                        "image_path": str(image_path.relative_to(DR_ROOT)),
                        "label": str(label),
                        "label_name": label_name_bidr(label),
                        "source_id": image_path.name,
                        "source_split": source_split,
                        "diagnosis": "",
                    }
                )
    return rows


def read_messidor2() -> list[dict[str, str]]:
    root = DR_ROOT / "Messidor-2 Dataset"
    image_dir = root / "messidor-2" / "messidor-2" / "preprocess"
    csv_path = root / "messidor_data.csv"
    rows: list[dict[str, str]] = []

    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            diagnosis = int(row["diagnosis"])
            label = 0 if diagnosis == 0 else 1
            image_path = image_dir / row["id_code"]
            rows.append(
                {
                    "dataset": "Messidor2",
                    "task": "binary",
                    "num_classes": "2",
                    "image_variant": "preprocess",
                    "image_path": str(image_path.relative_to(DR_ROOT)),
                    "label": str(label),
                    "label_name": label_name_messidor(label),
                    "source_id": row["id_code"],
                    "source_split": "",
                    "diagnosis": str(diagnosis),
                }
            )
    return rows


def split_rows(rows: list[dict[str, str]], seed: int = SEED) -> list[dict[str, str]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)

    output: list[dict[str, str]] = []
    for label in sorted(grouped, key=int):
        group = grouped[label]
        rng.shuffle(group)
        counts = split_counts(len(group))
        start = 0
        for split in RATIOS:
            end = start + counts[split]
            for row in group[start:end]:
                output.append({"split": split, **row})
            start = end
    return output


def validate(rows: list[dict[str, str]]) -> None:
    missing = [row["image_path"] for row in rows if not (DR_ROOT / row["image_path"]).exists()]
    if missing:
        examples = ", ".join(missing[:5])
        raise FileNotFoundError(f"{len(missing)} DR manifest images are missing, examples: {examples}")

    ids = [(row["dataset"], row["source_id"], row["source_split"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate DR dataset/source_id/source_split tuple found")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "dataset",
        "task",
        "num_classes",
        "image_variant",
        "image_path",
        "label",
        "label_name",
        "source_id",
        "source_split",
        "diagnosis",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_dataset(name: str, rows: list[dict[str, str]], out_dir: Path = OUT_DIR) -> None:
    dataset_dir = out_dir / name
    all_rows: list[dict[str, str]] = []
    for split in RATIOS:
        split_part = [row for row in rows if row["split"] == split]
        split_part.sort(key=lambda row: (int(row["label"]), row["source_id"]))
        all_rows.extend(split_part)
        write_csv(dataset_dir / f"{split}.csv", split_part)
    write_csv(dataset_dir / "all.csv", all_rows)


def print_counts(name: str, rows: list[dict[str, str]]) -> None:
    print(name)
    for split in RATIOS:
        split_part = [row for row in rows if row["split"] == split]
        counts = defaultdict(int)
        for row in split_part:
            counts[row["label_name"]] += 1
        print(" ", split, len(split_part), dict(sorted(counts.items())))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DR dataset manifests.")
    parser.add_argument("--dr-root", default=None, help="Folder containing BiDR and Messidor-2 Dataset. Overrides auto discovery and DR_DATA_ROOT.")
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="Manifest output directory.")
    parser.add_argument("--seed", type=int, default=SEED, help="Stratified split seed for Messidor2.")
    args = parser.parse_args()

    global DR_ROOT
    DR_ROOT = find_dr_root(args.dr_root)
    out_dir = Path(args.out_dir)

    bidr_rows = read_bidr()
    messidor_rows = split_rows(read_messidor2(), seed=args.seed)
    validate(bidr_rows + messidor_rows)

    write_dataset("BiDR", bidr_rows, out_dir=out_dir)
    write_dataset("Messidor2", messidor_rows, out_dir=out_dir)

    print(f"DR root: {DR_ROOT}")
    print("manifest image_path root: DR root")
    print(f"wrote {out_dir}")
    print_counts("BiDR", bidr_rows)
    print_counts("Messidor2", messidor_rows)


if __name__ == "__main__":
    main()
