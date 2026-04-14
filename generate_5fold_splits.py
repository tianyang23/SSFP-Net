#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate random 5-fold cross-validation txt splits for 3D-IRCADb-style patient folders.

Features:
- Read patient folder names from a base directory (e.g. patient_1 ~ patient_20)
- Random 5-fold split controlled by a seed
- For each fold:
  - test: 1 fold
  - remaining 4 folds -> choose a validation subset
  - rest -> train
- Within train:
  - generate labeled/unlabeled txt files for 50% and 20%
- 20% labeled is a subset of 50% labeled for reproducibility

Example:
python generate_5fold_splits.py \
  --base_dir /home/gpuserver/zhz/Datasets-3/3Dircadb/fold \
  --output_dir /home/gpuserver/zhz/Segtest_2/3D/new/xiangsidu/3Dircadb_5fold_splits_ssl_seed42 \
  --seed 42
"""

import os
import json
import math
import random
import argparse
from pathlib import Path


def list_patients(base_dir: str, prefix: str = "patient_"):
    base = Path(base_dir)

    if not base.is_dir():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")


    patients = []

    for p in base.iterdir():
        if p.is_dir() and p.name.startswith(prefix):
            patients.append(p.name)


    def sort_key(name: str):
        # patient_1, patient_2, ..., patient_20
        try:
            return int(name.split("_")[-1])
        except Exception:

            return name


    patients.sort(key=sort_key)
    return patients


def write_txt(path: Path, items):

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(f"{item}\n")


def split_k_folds(patients, n_folds=5):
    n = len(patients)

    base = n // n_folds

    rem = n % n_folds
    folds = []
    start = 0
    for i in range(n_folds):
        fold_size = base + (1 if i < rem else 0)
        folds.append(patients[start:start + fold_size])
        start += fold_size
    return folds


# Split training cases into labeled and unlabeled subsets.
def sample_labeled(train_patients, ratio_percent: int, rng: random.Random):
    n_train = len(train_patients)

    if ratio_percent == 50:
        n_labeled = max(1, round(n_train * 0.50))

    elif ratio_percent == 20:
        n_labeled = max(1, round(n_train * 0.20))
    else:
        raise ValueError("ratio_percent must be 20 or 50")


    labeled = sorted(rng.sample(train_patients, n_labeled), key=lambda x: int(x.split("_")[-1]))

    unlabeled = [p for p in train_patients if p not in labeled]
    return labeled, unlabeled


def main():
    parser = argparse.ArgumentParser(description="Generate random 5-fold txt splits")

    parser.add_argument("--base_dir", type=str, required=True,
                        help="Directory containing patient folders, e.g. patient_1 ~ patient_20")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save generated txt split files")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed controlling the 5-fold split")

    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of folds, default=5")

    parser.add_argument("--val_size", type=int, default=2,
                        help="Validation cases selected from the non-test cases in each fold, default=2")

    parser.add_argument("--prefix", type=str, default="patient_",
                        help="Patient folder prefix, default='patient_'")
    args = parser.parse_args()


    rng = random.Random(args.seed)

    patients = list_patients(args.base_dir, args.prefix)

    if len(patients) < args.n_folds:
        raise ValueError(f"Not enough patients ({len(patients)}) for {args.n_folds} folds")


    shuffled = patients[:]
    rng.shuffle(shuffled)


    folds = split_k_folds(shuffled, args.n_folds)
    out_dir = Path(args.output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # summary
    summary = {
        "base_dir": args.base_dir,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "n_folds": args.n_folds,
        "val_size": args.val_size,
        "patients": patients,
        "shuffled_patients": shuffled,
        "folds": {}
    }


    for i in range(args.n_folds):
        fold_id = i + 1

        test_patients = sorted(folds[i], key=lambda x: int(x.split("_")[-1]))
        remaining = []

        for j in range(args.n_folds):
            if j != i:
                remaining.extend(folds[j])

        # Use a fold-specific RNG so each fold remains reproducible under the same seed


        fold_rng = random.Random(args.seed * 100 + fold_id)


        if args.val_size >= len(remaining):
            raise ValueError(f"val_size={args.val_size} is too large for remaining={len(remaining)}")


        val_patients = sorted(fold_rng.sample(remaining, args.val_size), key=lambda x: int(x.split("_")[-1]))

        train_patients = sorted([p for p in remaining if p not in val_patients], key=lambda x: int(x.split("_")[-1]))


        labeled_50, unlabeled_50 = sample_labeled(train_patients, 50, fold_rng)

        # Make 20% labeled a subset of 50% labeled

        n_20 = max(1, round(len(train_patients) * 0.20))
        labeled_20 = sorted(fold_rng.sample(labeled_50, min(n_20, len(labeled_50))),
                            key=lambda x: int(x.split("_")[-1]))

        unlabeled_20 = [p for p in train_patients if p not in labeled_20]


        write_txt(out_dir / f"fold{fold_id}_train.txt", train_patients)
        write_txt(out_dir / f"fold{fold_id}_val.txt", val_patients)
        write_txt(out_dir / f"fold{fold_id}_test.txt", test_patients)


        write_txt(out_dir / f"fold{fold_id}_train_labeled_50.txt", labeled_50)
        write_txt(out_dir / f"fold{fold_id}_train_unlabeled_50.txt", unlabeled_50)
        write_txt(out_dir / f"fold{fold_id}_train_labeled_20.txt", labeled_20)
        write_txt(out_dir / f"fold{fold_id}_train_unlabeled_20.txt", unlabeled_20)


        summary["folds"][f"fold{fold_id}"] = {
            "train": train_patients,
            "val": val_patients,
            "test": test_patients,
            "train_labeled_50": labeled_50,
            "train_unlabeled_50": unlabeled_50,
            "train_labeled_20": labeled_20,
            "train_unlabeled_20": unlabeled_20,
            "counts": {
                "train": len(train_patients),
                "val": len(val_patients),
                "test": len(test_patients),
                "train_labeled_50": len(labeled_50),
                "train_unlabeled_50": len(unlabeled_50),
                "train_labeled_20": len(labeled_20),
                "train_unlabeled_20": len(unlabeled_20),
            }
        }


    with open(out_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


    with open(out_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write(
            "Random 5-fold split files generated by generate_5fold_splits.py\n"
            f"Seed: {args.seed}\n"
            f"Base dir: {args.base_dir}\n"
            f"Output dir: {args.output_dir}\n"
            f"Validation size per fold: {args.val_size}\n"
            "Files:\n"
            "  foldX_train.txt\n"
            "  foldX_val.txt\n"
            "  foldX_test.txt\n"
            "  foldX_train_labeled_50.txt\n"
            "  foldX_train_unlabeled_50.txt\n"
            "  foldX_train_labeled_20.txt\n"
            "  foldX_train_unlabeled_20.txt\n"
        )


    print(f"Generated random {args.n_folds}-fold split files in: {out_dir}")
    print(f"Seed: {args.seed}")


if __name__ == "__main__":
    main()
