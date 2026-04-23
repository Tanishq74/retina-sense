#!/usr/bin/env python3
"""
prepare_datasets.py — Dataset preparation utility for RetinaSense-ViT.

Downloads, preprocesses, and merges new datasets (EyePACS, REFUGE2, ADAM,
MESSIDOR-2) alongside existing APTOS + ODIR data.

Usage:
    python prepare_datasets.py --dataset eyepacs   --raw-dir ./data/eyepacs
    python prepare_datasets.py --dataset refuge     --raw-dir ./data/refuge
    python prepare_datasets.py --dataset adam        --raw-dir ./data/adam
    python prepare_datasets.py --dataset messidor2  --raw-dir ./data/messidor2
    python prepare_datasets.py --merge
    python prepare_datasets.py --status
"""

import argparse
import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(BASE_DIR, "preprocessed_cache_unified")
COMBINED_CSV = os.path.join(DATA_DIR, "combined_dataset.csv")
MERGED_CSV = os.path.join(DATA_DIR, "combined_dataset_merged.csv")

DISEASE_NAMES = {0: "Normal", 1: "DR", 2: "Glaucoma", 3: "Cataract", 4: "AMD"}
DATASET_NAMES = ["eyepacs", "refuge", "adam", "messidor2"]

DOWNLOAD_HINTS = {
    "eyepacs": (
        "EyePACS (~35K images, DR grades 0-4)\n"
        "  Download from Kaggle: kaggle competitions download -c diabetic-retinopathy-detection\n"
        "  Extract to: ./data/eyepacs/\n"
        "  Expected structure:\n"
        "    data/eyepacs/trainLabels.csv\n"
        "    data/eyepacs/train/<image_id>.jpeg"
    ),
    "refuge": (
        "REFUGE2 (~2000 images, Glaucoma + Normal)\n"
        "  Download from: https://refuge.grand-challenge.org/\n"
        "  Extract to: ./data/refuge/\n"
        "  Expected structure:\n"
        "    data/refuge/Training400/Glaucoma/*.jpg\n"
        "    data/refuge/Training400/Non-Glaucoma/*.jpg"
    ),
    "adam": (
        "ADAM (~1200 images, AMD + Normal)\n"
        "  Download from: https://amd.grand-challenge.org/\n"
        "  Extract to: ./data/adam/\n"
        "  Expected structure:\n"
        "    data/adam/Training400/AMD/*.jpg\n"
        "    data/adam/Training400/Non-AMD/*.jpg"
    ),
    "messidor2": (
        "MESSIDOR-2 (~1748 images, DR grades 0-4)\n"
        "  Download from Kaggle: kaggle datasets download -d google-brain/messidor2-dr-grades\n"
        "  Extract to: ./data/messidor2/\n"
        "  Expected structure:\n"
        "    data/messidor2/<labels>.csv  (with 'image' and 'adjudicated_dr_grade' columns)\n"
        "    data/messidor2/images/<image_id>.jpg (or .png, .tif)"
    ),
}

# ---------------------------------------------------------------------------
# Unified CLAHE preprocessing
# ---------------------------------------------------------------------------

def unified_clahe(path: str, sz: int = 224) -> Optional[np.ndarray]:
    """Apply unified CLAHE preprocessing with circular mask.

    Matches the pipeline used for APTOS + ODIR in the training set so that
    all datasets share a common domain.
    """
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (sz, sz), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (sz // 2, sz // 2), int(sz * 0.48), 255, -1)
    return cv2.bitwise_and(img, img, mask=mask)


# Try to import from unified_preprocessing module if it exists; fall back to
# the inline implementation above.
try:
    from unified_preprocessing import unified_clahe as _imported_clahe  # type: ignore
    unified_clahe = _imported_clahe  # noqa: F811
    print("[INFO] Using unified_clahe from unified_preprocessing module.")
except ImportError:
    pass


def preprocess_and_cache(src_path: str, source_tag: str) -> Optional[str]:
    """Run unified CLAHE on *src_path* and save the result to the cache.

    Returns the **relative** path (from BASE_DIR) of the cached .npy file,
    or None if the source image could not be read.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Deterministic cache key based on absolute path
    abs_path = os.path.abspath(src_path)
    path_hash = hashlib.md5(abs_path.encode()).hexdigest()[:12]
    stem = Path(src_path).stem
    cache_name = f"{source_tag}_{stem}_{path_hash}.npy"
    cache_path = os.path.join(CACHE_DIR, cache_name)
    rel_cache = os.path.relpath(cache_path, BASE_DIR)

    if os.path.exists(cache_path):
        return rel_cache

    img = unified_clahe(abs_path)
    if img is None:
        return None
    np.save(cache_path, img)
    return rel_cache


# ---------------------------------------------------------------------------
# Dataset handlers
# ---------------------------------------------------------------------------

def _find_images(directory: str, exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")) -> list:
    """Recursively find images under *directory*."""
    found = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(exts):
                found.append(os.path.join(root, f))
    return sorted(found)


def prepare_eyepacs(raw_dir: str) -> pd.DataFrame:
    """Prepare EyePACS dataset.

    Expects:
        raw_dir/trainLabels.csv   — columns: image, level
        raw_dir/train/*.jpeg
    """
    raw_dir = os.path.abspath(raw_dir)

    # Locate CSV
    csv_candidates = [
        os.path.join(raw_dir, "trainLabels.csv"),
        os.path.join(raw_dir, "trainLabels.csv.zip"),
    ]
    csv_path = None
    for c in csv_candidates:
        if os.path.isfile(c):
            csv_path = c
            break
    if csv_path is None:
        print(f"[ERROR] Cannot find trainLabels.csv in {raw_dir}")
        print(f"  Hint:\n{DOWNLOAD_HINTS['eyepacs']}")
        return pd.DataFrame()

    labels = pd.read_csv(csv_path)
    if "image" not in labels.columns or "level" not in labels.columns:
        print(f"[ERROR] trainLabels.csv must have 'image' and 'level' columns. Found: {list(labels.columns)}")
        return pd.DataFrame()

    # Locate image directory
    img_dir_candidates = [
        os.path.join(raw_dir, "train"),
        os.path.join(raw_dir, "train_images"),
        raw_dir,
    ]
    img_dir = None
    for d in img_dir_candidates:
        if os.path.isdir(d):
            # Quick check: does it contain at least one jpeg?
            sample = [f for f in os.listdir(d) if f.lower().endswith((".jpeg", ".jpg", ".png"))]
            if sample:
                img_dir = d
                break
    if img_dir is None:
        print(f"[ERROR] Cannot locate image directory under {raw_dir}")
        print(f"  Hint:\n{DOWNLOAD_HINTS['eyepacs']}")
        return pd.DataFrame()

    records = []
    skipped = 0
    for _, row in tqdm(labels.iterrows(), total=len(labels), desc="EyePACS"):
        img_id = str(row["image"])
        level = int(row["level"])

        # Find actual file (try common extensions)
        src = None
        for ext in (".jpeg", ".jpg", ".png"):
            candidate = os.path.join(img_dir, img_id + ext)
            if os.path.isfile(candidate):
                src = candidate
                break
        if src is None:
            skipped += 1
            continue

        # Map labels
        disease_label = 0 if level == 0 else 1  # Normal vs DR
        severity_label = level  # 0-4

        cached = preprocess_and_cache(src, "EYEPACS")
        if cached is None:
            skipped += 1
            continue

        records.append({
            "image_path": cached,
            "source": "EYEPACS",
            "disease_label": disease_label,
            "severity_label": severity_label,
        })

    if skipped:
        print(f"  [WARN] Skipped {skipped} images (missing file or unreadable)")

    df = pd.DataFrame(records)
    print(f"  Prepared {len(df)} EyePACS images.")
    return df


def prepare_refuge(raw_dir: str) -> pd.DataFrame:
    """Prepare REFUGE2 dataset.

    Looks for Glaucoma / Non-Glaucoma directory structures.
    """
    raw_dir = os.path.abspath(raw_dir)
    if not os.path.isdir(raw_dir):
        print(f"[ERROR] Directory not found: {raw_dir}")
        print(f"  Hint:\n{DOWNLOAD_HINTS['refuge']}")
        return pd.DataFrame()

    # Search patterns for glaucoma / non-glaucoma directories
    glaucoma_dirs = []
    non_glaucoma_dirs = []
    for root, dirs, _ in os.walk(raw_dir):
        for d in dirs:
            dl = d.lower()
            full = os.path.join(root, d)
            if dl in ("glaucoma", "glaucoma_images", "glaucoma_positive"):
                glaucoma_dirs.append(full)
            elif dl in ("non-glaucoma", "nonglaucoma", "non_glaucoma",
                        "normal", "non-glaucoma_images", "glaucoma_negative"):
                non_glaucoma_dirs.append(full)

    if not glaucoma_dirs and not non_glaucoma_dirs:
        print(f"[ERROR] Cannot find Glaucoma / Non-Glaucoma directories under {raw_dir}")
        print(f"  Searched recursively for directories named Glaucoma, Non-Glaucoma, etc.")
        print(f"  Hint:\n{DOWNLOAD_HINTS['refuge']}")
        return pd.DataFrame()

    records = []
    skipped = 0

    # Glaucoma images
    for gdir in glaucoma_dirs:
        imgs = _find_images(gdir)
        for src in tqdm(imgs, desc=f"REFUGE2 Glaucoma ({os.path.basename(gdir)})"):
            cached = preprocess_and_cache(src, "REFUGE2")
            if cached is None:
                skipped += 1
                continue
            records.append({
                "image_path": cached,
                "source": "REFUGE2",
                "disease_label": 2,
                "severity_label": -1,
            })

    # Non-Glaucoma images
    for ngdir in non_glaucoma_dirs:
        imgs = _find_images(ngdir)
        for src in tqdm(imgs, desc=f"REFUGE2 Normal ({os.path.basename(ngdir)})"):
            cached = preprocess_and_cache(src, "REFUGE2")
            if cached is None:
                skipped += 1
                continue
            records.append({
                "image_path": cached,
                "source": "REFUGE2",
                "disease_label": 0,
                "severity_label": -1,
            })

    if skipped:
        print(f"  [WARN] Skipped {skipped} images (unreadable)")

    df = pd.DataFrame(records)
    print(f"  Prepared {len(df)} REFUGE2 images.")
    return df


def prepare_adam(raw_dir: str) -> pd.DataFrame:
    """Prepare ADAM dataset.

    Looks for AMD / Non-AMD directory structures.
    """
    raw_dir = os.path.abspath(raw_dir)
    if not os.path.isdir(raw_dir):
        print(f"[ERROR] Directory not found: {raw_dir}")
        print(f"  Hint:\n{DOWNLOAD_HINTS['adam']}")
        return pd.DataFrame()

    amd_dirs = []
    non_amd_dirs = []
    for root, dirs, _ in os.walk(raw_dir):
        for d in dirs:
            dl = d.lower()
            full = os.path.join(root, d)
            if dl in ("amd", "amd_images", "amd_positive"):
                amd_dirs.append(full)
            elif dl in ("non-amd", "nonamd", "non_amd", "normal",
                        "non-amd_images", "amd_negative"):
                non_amd_dirs.append(full)

    if not amd_dirs and not non_amd_dirs:
        print(f"[ERROR] Cannot find AMD / Non-AMD directories under {raw_dir}")
        print(f"  Hint:\n{DOWNLOAD_HINTS['adam']}")
        return pd.DataFrame()

    records = []
    skipped = 0

    for adir in amd_dirs:
        imgs = _find_images(adir)
        for src in tqdm(imgs, desc=f"ADAM AMD ({os.path.basename(adir)})"):
            cached = preprocess_and_cache(src, "ADAM")
            if cached is None:
                skipped += 1
                continue
            records.append({
                "image_path": cached,
                "source": "ADAM",
                "disease_label": 4,
                "severity_label": -1,
            })

    for nadir in non_amd_dirs:
        imgs = _find_images(nadir)
        for src in tqdm(imgs, desc=f"ADAM Normal ({os.path.basename(nadir)})"):
            cached = preprocess_and_cache(src, "ADAM")
            if cached is None:
                skipped += 1
                continue
            records.append({
                "image_path": cached,
                "source": "ADAM",
                "disease_label": 0,
                "severity_label": -1,
            })

    if skipped:
        print(f"  [WARN] Skipped {skipped} images (unreadable)")

    df = pd.DataFrame(records)
    print(f"  Prepared {len(df)} ADAM images.")
    return df


def prepare_messidor2(raw_dir: str) -> pd.DataFrame:
    """Prepare MESSIDOR-2 dataset.

    Expects a CSV with 'image' and 'adjudicated_dr_grade' columns.
    """
    raw_dir = os.path.abspath(raw_dir)
    if not os.path.isdir(raw_dir):
        print(f"[ERROR] Directory not found: {raw_dir}")
        print(f"  Hint:\n{DOWNLOAD_HINTS['messidor2']}")
        return pd.DataFrame()

    # Find CSV
    csv_path = None
    for f in os.listdir(raw_dir):
        if f.lower().endswith(".csv"):
            candidate = os.path.join(raw_dir, f)
            try:
                tmp = pd.read_csv(candidate, nrows=5)
                if "adjudicated_dr_grade" in tmp.columns:
                    csv_path = candidate
                    break
            except Exception:
                continue

    if csv_path is None:
        print(f"[ERROR] Cannot find CSV with 'adjudicated_dr_grade' column in {raw_dir}")
        print(f"  Hint:\n{DOWNLOAD_HINTS['messidor2']}")
        return pd.DataFrame()

    labels = pd.read_csv(csv_path)
    print(f"  Found labels CSV: {csv_path} ({len(labels)} rows)")

    # Locate image directory
    img_dir_candidates = [
        os.path.join(raw_dir, "images"),
        os.path.join(raw_dir, "IMAGES"),
        os.path.join(raw_dir, "messidor_2"),
        raw_dir,
    ]
    img_dir = None
    for d in img_dir_candidates:
        if os.path.isdir(d):
            sample = _find_images(d)
            if sample:
                img_dir = d
                break

    if img_dir is None:
        print(f"[ERROR] Cannot locate image directory under {raw_dir}")
        return pd.DataFrame()

    # Build a lookup of available images by stem
    available = {}
    for fpath in _find_images(img_dir):
        stem = Path(fpath).stem.lower()
        available[stem] = fpath

    records = []
    skipped = 0

    image_col = "image" if "image" in labels.columns else labels.columns[0]

    for _, row in tqdm(labels.iterrows(), total=len(labels), desc="MESSIDOR-2"):
        img_id = str(row[image_col])
        grade = int(row["adjudicated_dr_grade"])

        # Try to find the image
        stem = Path(img_id).stem.lower()
        src = available.get(stem)
        if src is None:
            # Also try the raw value
            candidate = os.path.join(img_dir, img_id)
            if os.path.isfile(candidate):
                src = candidate
        if src is None:
            skipped += 1
            continue

        disease_label = 0 if grade == 0 else 1
        severity_label = grade

        cached = preprocess_and_cache(src, "MESSIDOR2")
        if cached is None:
            skipped += 1
            continue

        records.append({
            "image_path": cached,
            "source": "MESSIDOR2",
            "disease_label": disease_label,
            "severity_label": severity_label,
        })

    if skipped:
        print(f"  [WARN] Skipped {skipped} images (missing file or unreadable)")

    df = pd.DataFrame(records)
    print(f"  Prepared {len(df)} MESSIDOR-2 images.")
    return df


# ---------------------------------------------------------------------------
# Handler dispatch
# ---------------------------------------------------------------------------

HANDLERS = {
    "eyepacs": prepare_eyepacs,
    "refuge": prepare_refuge,
    "adam": prepare_adam,
    "messidor2": prepare_messidor2,
}


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

def merge_datasets() -> None:
    """Merge all available dataset CSVs into unified train/calib/test splits."""
    os.makedirs(DATA_DIR, exist_ok=True)

    frames = []

    # 1. Load existing combined_dataset.csv
    if os.path.isfile(COMBINED_CSV):
        existing = pd.read_csv(COMBINED_CSV)
        # Normalise column names — the original CSV uses 'dataset' instead of 'source'
        if "dataset" in existing.columns and "source" not in existing.columns:
            existing = existing.rename(columns={"dataset": "source"})
        required = {"image_path", "source", "disease_label", "severity_label"}
        if not required.issubset(set(existing.columns)):
            print(f"[WARN] {COMBINED_CSV} missing columns {required - set(existing.columns)}; skipping.")
        else:
            frames.append(existing)
            print(f"  Loaded existing dataset: {COMBINED_CSV} ({len(existing)} rows)")
    else:
        print(f"  [INFO] No existing {COMBINED_CSV} found; starting from scratch.")

    # 2. Load new dataset_*.csv files
    for name in DATASET_NAMES:
        csv_path = os.path.join(DATA_DIR, f"dataset_{name}.csv")
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
            frames.append(df)
            print(f"  Loaded new dataset: {csv_path} ({len(df)} rows)")

    if not frames:
        print("[ERROR] No datasets found to merge. Run --dataset first or ensure combined_dataset.csv exists.")
        return

    merged = pd.concat(frames, ignore_index=True)

    # Deduplicate by image_path (keep last — new data overrides old)
    before = len(merged)
    merged = merged.drop_duplicates(subset="image_path", keep="last").reset_index(drop=True)
    dupes = before - len(merged)
    if dupes:
        print(f"  Removed {dupes} duplicate image_path entries.")

    # Verify paths exist
    missing = []
    for idx, row in merged.iterrows():
        abs_path = os.path.join(BASE_DIR, row["image_path"]) if not os.path.isabs(row["image_path"]) else row["image_path"]
        if not os.path.exists(abs_path):
            missing.append(idx)
    if missing:
        print(f"  [WARN] {len(missing)} image paths do not exist on disk; removing them.")
        merged = merged.drop(missing).reset_index(drop=True)

    print(f"\n  Total merged images: {len(merged)}")
    _print_class_distribution(merged, indent=2)

    # Stratified 70/15/15 split
    print("\n  Performing stratified 70/15/15 split...")
    train_df, temp_df = train_test_split(
        merged, test_size=0.30, random_state=42,
        stratify=merged["disease_label"],
    )
    calib_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42,
        stratify=temp_df["disease_label"],
    )

    # Save
    merged.to_csv(MERGED_CSV, index=False)
    train_df.to_csv(os.path.join(DATA_DIR, "train_split.csv"), index=False)
    calib_df.to_csv(os.path.join(DATA_DIR, "calib_split.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, "test_split.csv"), index=False)

    print(f"\n  Saved:")
    print(f"    {MERGED_CSV}  ({len(merged)} rows)")
    print(f"    data/train_split.csv  ({len(train_df)} rows)")
    print(f"    data/calib_split.csv  ({len(calib_df)} rows)")
    print(f"    data/test_split.csv   ({len(test_df)} rows)")

    # Per-split class counts
    for label, split_df in [("Train", train_df), ("Calib", calib_df), ("Test", test_df)]:
        counts = split_df["disease_label"].value_counts().sort_index()
        summary = ", ".join(f"{DISEASE_NAMES.get(k, k)}: {v}" for k, v in counts.items())
        print(f"    {label}: {summary}")


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def print_status() -> None:
    """Print an overview of all available datasets and their sizes."""
    print("=" * 65)
    print("  RetinaSense-ViT Dataset Status")
    print("=" * 65)

    # Existing combined dataset
    if os.path.isfile(COMBINED_CSV):
        df = pd.read_csv(COMBINED_CSV)
        print(f"\n  [EXISTING] {COMBINED_CSV}")
        print(f"    Rows: {len(df)}")
        _print_class_distribution(df, indent=4)
    else:
        print(f"\n  [EXISTING] {COMBINED_CSV} — NOT FOUND")

    # Per-dataset CSVs
    for name in DATASET_NAMES:
        csv_path = os.path.join(DATA_DIR, f"dataset_{name}.csv")
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
            print(f"\n  [NEW] dataset_{name}.csv")
            print(f"    Rows: {len(df)}")
            _print_class_distribution(df, indent=4)
        else:
            print(f"\n  [NEW] dataset_{name}.csv — not yet prepared")
            print(f"    {DOWNLOAD_HINTS[name].splitlines()[0]}")

    # Merged
    if os.path.isfile(MERGED_CSV):
        df = pd.read_csv(MERGED_CSV)
        print(f"\n  [MERGED] {MERGED_CSV}")
        print(f"    Rows: {len(df)}")
        _print_class_distribution(df, indent=4)
    else:
        print(f"\n  [MERGED] Not yet created. Run: python prepare_datasets.py --merge")

    # Splits
    for split_name in ("train_split", "calib_split", "test_split"):
        sp = os.path.join(DATA_DIR, f"{split_name}.csv")
        if os.path.isfile(sp):
            sdf = pd.read_csv(sp)
            print(f"    {split_name}.csv: {len(sdf)} rows")

    # Cache
    if os.path.isdir(CACHE_DIR):
        n_cached = len([f for f in os.listdir(CACHE_DIR) if f.endswith(".npy")])
        print(f"\n  Preprocessed cache: {CACHE_DIR} ({n_cached} files)")
    else:
        print(f"\n  Preprocessed cache: {CACHE_DIR} — not yet created")

    print("=" * 65)


def _print_class_distribution(df: pd.DataFrame, indent: int = 2) -> None:
    """Print class distribution of a dataframe."""
    prefix = " " * indent

    # By disease label
    label_col = "disease_label"
    if label_col not in df.columns:
        return
    counts = df[label_col].value_counts().sort_index()
    print(f"{prefix}Class distribution:")
    for label, count in counts.items():
        name = DISEASE_NAMES.get(int(label), f"Unknown({label})")
        pct = 100.0 * count / len(df)
        print(f"{prefix}  {label} ({name:>10s}): {count:>6d}  ({pct:5.1f}%)")

    # By source
    if "source" in df.columns:
        src_counts = df["source"].value_counts().sort_index()
        print(f"{prefix}Sources: {dict(src_counts)}")
    elif "dataset" in df.columns:
        src_counts = df["dataset"].value_counts().sort_index()
        print(f"{prefix}Sources: {dict(src_counts)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare and merge datasets for RetinaSense-ViT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python prepare_datasets.py --dataset eyepacs --raw-dir ./data/eyepacs\n"
            "  python prepare_datasets.py --dataset refuge  --raw-dir ./data/refuge\n"
            "  python prepare_datasets.py --merge\n"
            "  python prepare_datasets.py --status\n"
        ),
    )
    parser.add_argument(
        "--dataset", type=str, choices=DATASET_NAMES,
        help="Dataset to prepare (eyepacs, refuge, adam, messidor2).",
    )
    parser.add_argument(
        "--raw-dir", type=str, default=None,
        help="Path to the raw (downloaded) dataset directory.",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Merge all available datasets into unified train/calib/test splits.",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print overview of available datasets and counts.",
    )
    parser.add_argument(
        "--img-size", type=int, default=224,
        help="Image size for preprocessing (default: 224).",
    )

    args = parser.parse_args()

    if not args.dataset and not args.merge and not args.status:
        parser.print_help()
        sys.exit(1)

    # -- Status mode --
    if args.status:
        print_status()
        return

    # -- Dataset preparation mode --
    if args.dataset:
        if not args.raw_dir:
            default_dir = os.path.join(DATA_DIR, args.dataset)
            if os.path.isdir(default_dir):
                args.raw_dir = default_dir
                print(f"[INFO] Using default raw directory: {args.raw_dir}")
            else:
                print(f"[ERROR] --raw-dir is required (or place data in {default_dir}).")
                print(f"  Hint:\n{DOWNLOAD_HINTS[args.dataset]}")
                sys.exit(1)

        handler = HANDLERS[args.dataset]
        print(f"\n{'='*50}")
        print(f"  Preparing: {args.dataset.upper()}")
        print(f"  Raw dir:   {os.path.abspath(args.raw_dir)}")
        print(f"{'='*50}\n")

        df = handler(args.raw_dir)

        if df.empty:
            print(f"\n[ERROR] No images were prepared for {args.dataset}.")
            sys.exit(1)

        # Save intermediate CSV
        os.makedirs(DATA_DIR, exist_ok=True)
        out_csv = os.path.join(DATA_DIR, f"dataset_{args.dataset}.csv")
        df.to_csv(out_csv, index=False)
        print(f"\n  Saved: {out_csv} ({len(df)} rows)")
        _print_class_distribution(df, indent=2)

    # -- Merge mode --
    if args.merge:
        print(f"\n{'='*50}")
        print(f"  Merging all datasets")
        print(f"{'='*50}\n")
        merge_datasets()


if __name__ == "__main__":
    main()
