#!/usr/bin/env python3
"""
unified_preprocessing.py — Unified CLAHE preprocessing for RetinaSense-ViT

Applies the SAME CLAHE pipeline to ALL fundus images regardless of source
(APTOS or ODIR), eliminating the domain shift caused by different preprocessing
strategies. Saves preprocessed images as .npy files in preprocessed_cache_unified/.

Usage:
    python unified_preprocessing.py                 # Rebuild full cache
    python unified_preprocessing.py --recompute-stats  # Only recompute norm stats
    python unified_preprocessing.py --verify           # Verify cache completeness
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from PIL import Image


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "preprocessed_cache_unified")
CONFIGS_DIR = os.path.join(BASE_DIR, "configs")
NORM_STATS_PATH = os.path.join(CONFIGS_DIR, "fundus_norm_stats_unified.json")
DEFAULT_SIZE = 224

CSV_PATHS = [
    os.path.join(BASE_DIR, "data", "combined_dataset.csv"),
    os.path.join(BASE_DIR, "data", "train_split.csv"),
    os.path.join(BASE_DIR, "data", "calib_split.csv"),
    os.path.join(BASE_DIR, "data", "test_split.csv"),
]


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------
def read_rgb(path: str) -> np.ndarray:
    """Read an image file and return as RGB numpy array.

    Tries OpenCV first, falls back to PIL for uncommon formats.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Fallback: PIL handles more formats (e.g. TIFF, WebP)
    pil_img = Image.open(path).convert("RGB")
    return np.array(pil_img)


def resolve_image_path(raw_path: str) -> str:
    """Resolve a potentially relative CSV image path to an absolute path.

    Strips leading './' or './/' and joins with BASE_DIR.
    """
    cleaned = raw_path
    # Strip leading .// or ./
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    abs_path = os.path.join(BASE_DIR, cleaned)
    return abs_path


def cache_key(image_path: str, sz: int = DEFAULT_SIZE) -> str:
    """Derive cache filename: {stem}_{sz}.npy"""
    stem = Path(image_path).stem
    return f"{stem}_{sz}.npy"


# ---------------------------------------------------------------------------
# Core preprocessing
# ---------------------------------------------------------------------------
def crop_black_borders(img: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Crop dark borders from a fundus image.

    Converts to grayscale, finds rows/cols whose mean intensity exceeds
    *threshold*, and crops to that bounding box. Returns the original image
    if no valid crop region is found.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Row means and column means
    row_means = gray.mean(axis=1)
    col_means = gray.mean(axis=0)

    rows_above = np.where(row_means > threshold)[0]
    cols_above = np.where(col_means > threshold)[0]

    if len(rows_above) == 0 or len(cols_above) == 0:
        return img  # Nothing to crop — return as-is

    y_min, y_max = rows_above[0], rows_above[-1]
    x_min, x_max = cols_above[0], cols_above[-1]

    # Guard: don't crop to something absurdly small
    if (y_max - y_min) < 10 or (x_max - x_min) < 10:
        return img

    return img[y_min : y_max + 1, x_min : x_max + 1]


def unified_clahe(path: str, sz: int = DEFAULT_SIZE) -> np.ndarray:
    """Single CLAHE pipeline for all sources. No domain-conditional branching.

    Steps:
        1. Read image as RGB
        2. Crop black borders (fundus images often have dark surrounds)
        3. Resize to (sz, sz)
        4. Apply CLAHE on L-channel in LAB colour space
        5. Apply circular mask to isolate fundus disc

    Returns:
        np.ndarray — preprocessed RGB image, uint8, shape (sz, sz, 3)
    """
    img = read_rgb(path)
    img = crop_black_borders(img)
    img = cv2.resize(img, (sz, sz), interpolation=cv2.INTER_AREA)

    # CLAHE on L-channel
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Circular mask
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (sz // 2, sz // 2), int(sz * 0.48), 255, -1)
    img = cv2.bitwise_and(img, img, mask=mask)

    return img


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------
def collect_all_image_paths() -> pd.DataFrame:
    """Load and deduplicate image entries from all CSV files.

    Returns a DataFrame with at least columns: image_path, source (if present).
    Training-split membership is tracked via an 'is_train' column.
    """
    frames = []
    train_paths_set = set()

    for csv_path in CSV_PATHS:
        if not os.path.isfile(csv_path):
            print(f"[WARN] CSV not found, skipping: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        if "image_path" not in df.columns:
            print(f"[WARN] No 'image_path' column in {csv_path}, skipping.")
            continue

        # Track training split membership
        basename = os.path.basename(csv_path).lower()
        if "train" in basename:
            for p in df["image_path"]:
                train_paths_set.add(resolve_image_path(str(p)))

        frames.append(df)

    if not frames:
        print("[ERROR] No valid CSVs found. Cannot proceed.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["image_path"], keep="first")
    combined["abs_path"] = combined["image_path"].apply(lambda p: resolve_image_path(str(p)))
    combined["is_train"] = combined["abs_path"].isin(train_paths_set)

    print(f"Collected {len(combined)} unique images from {len(frames)} CSV(s).")
    return combined


# ---------------------------------------------------------------------------
# Cache building
# ---------------------------------------------------------------------------
def build_cache(df: pd.DataFrame, sz: int = DEFAULT_SIZE) -> dict:
    """Preprocess all images and save to cache directory.

    Returns a summary dict with counts of total, newly_cached, skipped, failed.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    total = len(df)
    newly_cached = 0
    skipped = 0
    failed = 0
    failures = []

    for _, row in tqdm(df.iterrows(), total=total, desc="Preprocessing"):
        abs_path = row["abs_path"]
        key = cache_key(abs_path, sz)
        out_path = os.path.join(CACHE_DIR, key)

        # Skip if already cached
        if os.path.isfile(out_path):
            skipped += 1
            continue

        try:
            img = unified_clahe(abs_path, sz)
            np.save(out_path, img)
            newly_cached += 1
        except Exception as exc:
            failed += 1
            failures.append((abs_path, str(exc)))

    summary = {
        "total": total,
        "newly_cached": newly_cached,
        "already_cached": skipped,
        "failed": failed,
        "failures": failures,
    }
    return summary


# ---------------------------------------------------------------------------
# Norm stats computation
# ---------------------------------------------------------------------------
def compute_norm_stats(df: pd.DataFrame, sz: int = DEFAULT_SIZE) -> dict:
    """Compute per-channel mean and std over the training split only.

    Reads cached .npy files. Pixels outside the circular mask (all-zero) are
    excluded from statistics.

    Returns dict with 'mean_rgb' and 'std_rgb' (each a list of 3 floats).
    """
    train_df = df[df["is_train"]]
    if len(train_df) == 0:
        print("[WARN] No training-split images identified. Computing stats over ALL images.")
        train_df = df

    print(f"Computing norm stats over {len(train_df)} training images...")

    # Online Welford-style accumulation (two-pass for numerical stability)
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = np.float64(0)

    missing = 0
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Norm stats"):
        key = cache_key(row["abs_path"], sz)
        npy_path = os.path.join(CACHE_DIR, key)
        if not os.path.isfile(npy_path):
            missing += 1
            continue

        img = np.load(npy_path).astype(np.float64) / 255.0  # (H, W, 3)

        # Mask: only count non-black pixels (inside the circular mask)
        mask = img.sum(axis=2) > 0  # (H, W)
        n_pixels = mask.sum()
        if n_pixels == 0:
            continue

        for c in range(3):
            vals = img[:, :, c][mask]
            channel_sum[c] += vals.sum()
            channel_sq_sum[c] += (vals ** 2).sum()
        pixel_count += n_pixels

    if missing > 0:
        print(f"[WARN] {missing} cached files missing during norm stats computation.")

    if pixel_count == 0:
        print("[ERROR] No valid pixels found. Cannot compute stats.")
        return {"mean_rgb": [0.0, 0.0, 0.0], "std_rgb": [1.0, 1.0, 1.0]}

    mean_rgb = (channel_sum / pixel_count).tolist()
    std_rgb = np.sqrt(channel_sq_sum / pixel_count - np.array(mean_rgb) ** 2).tolist()

    stats = {"mean_rgb": mean_rgb, "std_rgb": std_rgb}
    return stats


def save_norm_stats(stats: dict) -> None:
    """Save norm stats to JSON config file."""
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    with open(NORM_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Norm stats saved to {NORM_STATS_PATH}")
    print(f"  mean_rgb: {[round(v, 4) for v in stats['mean_rgb']]}")
    print(f"  std_rgb:  {[round(v, 4) for v in stats['std_rgb']]}")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_cache(df: pd.DataFrame, sz: int = DEFAULT_SIZE) -> None:
    """Check that every image in the dataset has a corresponding cache file."""
    total = len(df)
    present = 0
    missing_files = []

    for _, row in tqdm(df.iterrows(), total=total, desc="Verifying cache"):
        key = cache_key(row["abs_path"], sz)
        npy_path = os.path.join(CACHE_DIR, key)
        if os.path.isfile(npy_path):
            present += 1
        else:
            missing_files.append(row["abs_path"])

    print(f"\nCache verification: {present}/{total} files present.")
    if missing_files:
        print(f"  {len(missing_files)} MISSING:")
        for p in missing_files[:20]:
            print(f"    - {p}")
        if len(missing_files) > 20:
            print(f"    ... and {len(missing_files) - 20} more.")
    else:
        print("  All cached. Cache is complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Unified CLAHE preprocessing for RetinaSense-ViT"
    )
    parser.add_argument(
        "--recompute-stats",
        action="store_true",
        help="Only recompute norm stats from existing cache (skip preprocessing).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify cache completeness without rebuilding.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help=f"Target image size (default: {DEFAULT_SIZE}).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("RetinaSense-ViT  —  Unified CLAHE Preprocessing")
    print("=" * 60)
    print(f"Base dir:   {BASE_DIR}")
    print(f"Cache dir:  {CACHE_DIR}")
    print(f"Image size: {args.size}")
    print()

    # Collect all image paths
    df = collect_all_image_paths()

    if args.verify:
        verify_cache(df, sz=args.size)
        return

    if args.recompute_stats:
        stats = compute_norm_stats(df, sz=args.size)
        save_norm_stats(stats)
        return

    # Full rebuild: preprocess + compute stats
    summary = build_cache(df, sz=args.size)

    print()
    print("-" * 40)
    print("Preprocessing Summary")
    print("-" * 40)
    print(f"  Total images:    {summary['total']}")
    print(f"  Newly cached:    {summary['newly_cached']}")
    print(f"  Already cached:  {summary['already_cached']}")
    print(f"  Failed:          {summary['failed']}")

    if summary["failures"]:
        print("\n  Failed files:")
        for path, err in summary["failures"][:20]:
            print(f"    - {path}")
            print(f"      Error: {err}")
        if len(summary["failures"]) > 20:
            print(f"    ... and {len(summary['failures']) - 20} more.")

    # Compute and save norm stats
    print()
    stats = compute_norm_stats(df, sz=args.size)
    save_norm_stats(stats)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
