#!/usr/bin/env python3
"""
RetinaSense — Unified Preprocessing Pipeline
=============================================
Replaces the domain-conditional preprocessing (Ben Graham for APTOS, CLAHE for ODIR)
with a single CLAHE pipeline for ALL images. This eliminates the domain shift that
caused the ViT to learn source-specific features instead of disease features.

Usage:
    # Rebuild the entire cache with unified preprocessing
    python unified_preprocessing.py

    # Or import and use in training scripts:
    from unified_preprocessing import unified_preprocess, rebuild_cache
"""

import os, json, sys
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm

TARGET_SIZE = 224
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _crop_black_borders(img, tol=7):
    """Remove dark border padding common in fundus images."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return img
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return img[rmin:rmax+1, cmin:cmax+1]


def _apply_circular_mask(img):
    """Zero out pixels outside the circular fundus field of view."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    r = int(min(h, w) * 0.48)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    return cv2.bitwise_and(img, img, mask=mask)


def unified_preprocess(img, target_size=TARGET_SIZE):
    """Unified CLAHE preprocessing for ALL images regardless of source.

    Pipeline: crop borders -> resize -> CLAHE on L-channel -> circular mask

    Args:
        img: RGB numpy array (any size)
        target_size: output size (default 224)

    Returns:
        Preprocessed uint8 RGB array of shape (target_size, target_size, 3)
    """
    # Step 1: Crop black borders
    img = _crop_black_borders(img)

    # Step 2: Resize
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # Step 3: CLAHE on L-channel (same params as ODIR training)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    img = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    # Step 4: Circular mask
    img = _apply_circular_mask(img)

    return np.clip(img, 0, 255).astype(np.uint8)


def unified_preprocess_from_path(image_path, target_size=TARGET_SIZE):
    """Load an image from disk and apply unified preprocessing."""
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return unified_preprocess(img, target_size)


def compute_norm_stats(cache_dir, csv_path=None):
    """Compute channel-wise mean and std from the preprocessed cache.

    Should be run AFTER rebuilding the cache with unified preprocessing
    to get updated normalization statistics.
    """
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    n_pixels = 0

    npy_files = [f for f in os.listdir(cache_dir) if f.endswith('.npy')]
    if csv_path:
        df = pd.read_csv(csv_path)
        npy_files = [os.path.basename(p) for p in df['cache_path'].values if os.path.exists(p)]

    for fname in tqdm(npy_files, desc='Computing norm stats'):
        fp = os.path.join(cache_dir, fname) if not os.path.isabs(fname) else fname
        if not os.path.exists(fp):
            continue
        img = np.load(fp).astype(np.float64) / 255.0
        pixel_sum += img.sum(axis=(0, 1))
        pixel_sq_sum += (img ** 2).sum(axis=(0, 1))
        n_pixels += img.shape[0] * img.shape[1]

    mean = pixel_sum / n_pixels
    std = np.sqrt(pixel_sq_sum / n_pixels - mean ** 2)

    return mean.tolist(), std.tolist()


def rebuild_cache(csv_path, cache_dir, target_size=TARGET_SIZE):
    """Rebuild the entire preprocessed cache using unified CLAHE preprocessing.

    Args:
        csv_path: Path to CSV with 'image_path' column
        cache_dir: Directory to save .npy cache files
        target_size: Image size (default 224)

    Returns:
        List of cache file paths
    """
    os.makedirs(cache_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    cache_paths = []
    processed = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Rebuilding cache'):
        image_path = row['image_path']
        stem = os.path.splitext(os.path.basename(image_path))[0]
        cache_fp = os.path.join(cache_dir, f'{stem}_{target_size}.npy')

        img = unified_preprocess_from_path(image_path, target_size)
        np.save(cache_fp, img)
        cache_paths.append(cache_fp)
        processed += 1

    print(f'  Processed {processed} images -> {cache_dir}/')
    return cache_paths


def main():
    """Rebuild cache and recompute normalization stats."""
    cache_dir = os.path.join(BASE_DIR, 'preprocessed_cache_unified')
    data_dir = os.path.join(BASE_DIR, 'data')

    # Check for data CSVs
    combined_csv = os.path.join(data_dir, 'combined_dataset.csv')
    train_csv = os.path.join(data_dir, 'train_split.csv')

    if not os.path.exists(train_csv):
        print('ERROR: data/train_split.csv not found.')
        print('This script requires the training data CSVs and raw images.')
        print('Run this on the GPU server where data is available.')
        sys.exit(1)

    print('=' * 60)
    print('  RetinaSense — Unified Preprocessing Pipeline')
    print('=' * 60)
    print(f'  Cache dir: {cache_dir}')
    print(f'  Target size: {TARGET_SIZE}x{TARGET_SIZE}')
    print()

    # Rebuild cache for all splits
    for split_name in ['train_split', 'calib_split', 'test_split']:
        csv_path = os.path.join(data_dir, f'{split_name}.csv')
        if os.path.exists(csv_path):
            print(f'\n  Processing {split_name}...')
            rebuild_cache(csv_path, cache_dir)

    # Recompute normalization stats
    print('\n  Computing new normalization statistics...')
    mean, std = compute_norm_stats(cache_dir)
    print(f'  New mean: {mean}')
    print(f'  New std:  {std}')

    # Save new norm stats
    stats_path = os.path.join(BASE_DIR, 'configs', 'fundus_norm_stats_unified.json')
    with open(stats_path, 'w') as f:
        json.dump({'mean_rgb': mean, 'std_rgb': std}, f, indent=2)
    print(f'  Saved: {stats_path}')

    # Update the CSV files with new cache paths
    for split_name in ['train_split', 'calib_split', 'test_split']:
        csv_path = os.path.join(data_dir, f'{split_name}.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['cache_path'] = df['image_path'].apply(
                lambda p: os.path.join(cache_dir,
                    f'{os.path.splitext(os.path.basename(p))[0]}_{TARGET_SIZE}.npy')
            )
            unified_csv = csv_path.replace('.csv', '_unified.csv')
            df.to_csv(unified_csv, index=False)
            print(f'  Saved: {unified_csv}')

    print('\n' + '=' * 60)
    print('  Done! Next steps:')
    print('  1. Update retinasense_v3.py to use preprocessed_cache_unified/')
    print('  2. Update configs/fundus_norm_stats.json with the new stats')
    print('  3. Retrain the model')
    print('=' * 60)


if __name__ == '__main__':
    main()
