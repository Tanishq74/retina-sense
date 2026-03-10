#!/usr/bin/env python3
"""
RetinaSense v3 — Domain-Conditional Preprocessing Pipeline
===========================================================
Implements source-aware preprocessing:
  - APTOS   -> Ben Graham enhancement (high contrast DR-specific pipeline)
  - ODIR    -> CLAHE only (preserves sharpness, normalizes contrast)
  - REFUGE2 -> Resize only (images already clinical-grade high quality)

Image path resolution:
  - ODIR:  odir/preprocessed_images/<filename>
  - APTOS: aptos/gaussian_filtered_images/gaussian_filtered_images/<class>/<id>.png
           (looked up from aptos/train.csv; aptos/train_images/ does NOT exist)

Cache format: ./preprocessed_cache_v3/<stem>_v3.npy
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings('ignore')

# =========================================================
# PATHS
# =========================================================
BASE_DIR   = '/teamspace/studios/this_studio'
CSV_PATH   = os.path.join(BASE_DIR, 'data', 'combined_dataset.csv')
CACHE_DIR  = os.path.join(BASE_DIR, 'preprocessed_cache_v3')
DATA_DIR   = os.path.join(BASE_DIR, 'data')

ODIR_IMG_DIR   = os.path.join(BASE_DIR, 'odir', 'preprocessed_images')
APTOS_CSV      = os.path.join(BASE_DIR, 'aptos', 'train.csv')
APTOS_IMG_BASE = os.path.join(BASE_DIR, 'aptos',
                               'gaussian_filtered_images',
                               'gaussian_filtered_images')
APTOS_DIAG_MAP = {0: 'No_DR', 1: 'Mild', 2: 'Moderate',
                  3: 'Severe', 4: 'Proliferate_DR'}

ODIR_SAMPLE    = os.path.join(BASE_DIR, 'ocular-disease-recognition-odir5k',
                               'preprocessed_images', '2977_left.jpg')

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR,  exist_ok=True)

TARGET_SIZE = 224

# =========================================================
# APTOS PATH LOOKUP TABLE
# Built once at module load; maps id_code (stem) -> abs path
# =========================================================

def _build_aptos_lookup() -> dict:
    """Return dict mapping aptos id_code -> absolute image path."""
    lookup = {}
    if not os.path.exists(APTOS_CSV):
        return lookup
    df = pd.read_csv(APTOS_CSV)
    for _, row in df.iterrows():
        folder = APTOS_DIAG_MAP.get(int(row['diagnosis']), 'No_DR')
        path   = os.path.join(APTOS_IMG_BASE, folder,
                              str(row['id_code']) + '.png')
        lookup[str(row['id_code'])] = path
    return lookup


_APTOS_LOOKUP: dict = _build_aptos_lookup()


# =========================================================
# PATH RESOLVER
# =========================================================

def resolve_image_path(raw_path: str, dataset: str = None) -> str:
    """
    Resolve CSV path entry to an absolute filesystem path.

    The CSV stores paths like:
      ODIR:  .//odir/preprocessed_images/0_left.jpg
      APTOS: .//aptos/train_images/000c1434d8d7.png  (train_images doesn't exist)

    Resolution rules:
      1. If the resolved path already exists, return it.
      2. ODIR: remap to odir/preprocessed_images/<filename>
      3. APTOS: look up via _APTOS_LOOKUP by stem
    """
    # Normalise .// and ./ prefixes
    p = raw_path.strip()
    if p.startswith('.//'):
        p = p[3:]
    elif p.startswith('./'):
        p = p[2:]

    # Try as-is (absolute or relative to BASE_DIR)
    if not os.path.isabs(p):
        candidate = os.path.join(BASE_DIR, p)
    else:
        candidate = p

    if os.path.exists(candidate):
        return candidate

    fname = os.path.basename(p)
    stem  = os.path.splitext(fname)[0]
    src   = (dataset or '').upper().strip()

    # ODIR remap
    if src == 'ODIR' or 'odir' in p.lower():
        return os.path.join(ODIR_IMG_DIR, fname)

    # APTOS remap via lookup table
    if src == 'APTOS' or 'aptos' in p.lower():
        if stem in _APTOS_LOOKUP:
            return _APTOS_LOOKUP[stem]

    # Final fallback: try all known image dirs
    for d in [ODIR_IMG_DIR, APTOS_IMG_BASE]:
        candidate2 = os.path.join(d, fname)
        if os.path.exists(candidate2):
            return candidate2

    return candidate  # return best guess even if missing


# =========================================================
# PREPROCESSING FUNCTIONS
# =========================================================

def _load_image(image_path: str):
    """Load image as RGB numpy array (H, W, 3) uint8. Returns None on failure."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _crop_black_borders(img: np.ndarray, tol: int = 7) -> np.ndarray:
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


def _apply_circular_mask(img: np.ndarray) -> np.ndarray:
    """Zero out pixels outside the circular fundus field of view."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    r = int(min(h, w) * 0.48)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    return cv2.bitwise_and(img, img, mask=mask)


def ben_graham_preprocess(img: np.ndarray, target_size: int = TARGET_SIZE,
                           sigma: float = 10.0) -> np.ndarray:
    """
    Ben Graham fundus enhancement — used for APTOS images.

    Enhances local retinal structures (vessels, lesions) by subtracting a
    Gaussian-blurred version from itself, centering intensity around 128.
    This removes low-frequency illumination variation (vignetting, uneven
    camera lighting) and amplifies high-frequency structural details.

    Formula: result = 4*img - 4*GaussianBlur(img, sigma=10) + 128
    Then circular mask applied to suppress black border.
    """
    img = _crop_black_borders(img)
    img = cv2.resize(img, (target_size, target_size),
                     interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    img  = cv2.addWeighted(img, 4, blur, -4, 128)
    img  = _apply_circular_mask(img)
    return np.clip(img, 0, 255).astype(np.uint8)


def clahe_preprocess(img: np.ndarray, target_size: int = TARGET_SIZE,
                     clip_limit: float = 2.0,
                     tile_grid: tuple = (8, 8)) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) — used for ODIR.

    Preserves image sharpness while normalizing local contrast.
    Applied only to the L (luminance) channel in LAB color space to
    avoid hue shifts. ODIR is a multi-source dataset with mixed quality,
    so CLAHE provides gentle contrast normalization without destroying
    fine detail the way Ben Graham's aggressive subtraction would.

    clip_limit=2.0: moderate clipping to prevent over-amplification of noise.
    tile_grid=(8,8): 8x8 tiles for local adaptation at appropriate scale.
    """
    img = _crop_black_borders(img)
    img = cv2.resize(img, (target_size, target_size),
                     interpolation=cv2.INTER_AREA)
    lab    = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe  = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq   = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    img    = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    img    = _apply_circular_mask(img)
    return np.clip(img, 0, 255).astype(np.uint8)


def resize_only_preprocess(img: np.ndarray,
                            target_size: int = TARGET_SIZE) -> np.ndarray:
    """
    Minimal preprocessing — used for REFUGE2.

    REFUGE2 images are acquired with a Zeiss Visucam 500 camera under
    standardized clinical conditions. They are already high-quality with
    consistent lighting. Any additional enhancement would degrade quality.
    """
    img = cv2.resize(img, (target_size, target_size),
                     interpolation=cv2.INTER_AREA)
    return np.clip(img, 0, 255).astype(np.uint8)


def preprocess_image(image_path: str, source: str,
                     target_size: int = TARGET_SIZE):
    """
    Domain-conditional preprocessing dispatcher.

    Parameters
    ----------
    image_path : str
        Absolute path to the fundus image file.
    source : str
        Dataset source. One of: 'APTOS', 'ODIR', 'REFUGE2' (case-insensitive).
    target_size : int
        Output spatial dimension (square). Default 224.

    Returns
    -------
    np.ndarray of shape (target_size, target_size, 3), dtype uint8,
    or None if the image cannot be loaded.
    """
    img = _load_image(image_path)
    if img is None:
        return None

    src = source.upper().strip()
    if src == 'APTOS':
        return ben_graham_preprocess(img, target_size=target_size)
    elif src == 'ODIR':
        return clahe_preprocess(img, target_size=target_size)
    elif src == 'REFUGE2':
        return resize_only_preprocess(img, target_size=target_size)
    else:
        # Safe fallback for unknown sources
        print(f'[WARN] Unknown source "{source}", applying CLAHE fallback.')
        return clahe_preprocess(img, target_size=target_size)


# =========================================================
# CACHE HELPERS
# =========================================================

def cache_path_for(raw_csv_path: str) -> str:
    """Return the .npy cache path for a given CSV image_path entry."""
    stem = Path(raw_csv_path).stem
    return os.path.join(CACHE_DIR, f'{stem}_v3.npy')


def is_cached(raw_csv_path: str) -> bool:
    return os.path.exists(cache_path_for(raw_csv_path))


def save_to_cache(raw_csv_path: str, arr: np.ndarray) -> None:
    np.save(cache_path_for(raw_csv_path), arr)


def load_from_cache(raw_csv_path: str):
    cp = cache_path_for(raw_csv_path)
    return np.load(cp) if os.path.exists(cp) else None


def cache_dataset(df: pd.DataFrame) -> dict:
    """
    Preprocess and cache all images in df using domain-conditional pipeline.
    Returns stats dict.
    """
    stats = dict(processed=0, skipped_missing=0, already_cached=0,
                 errors=0, total=len(df))

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Caching v3'):
        raw  = row['image_path']
        src  = row['dataset']

        if is_cached(raw):
            stats['already_cached'] += 1
            continue

        abs_path = resolve_image_path(raw, src)
        if not os.path.exists(abs_path):
            stats['skipped_missing'] += 1
            continue

        arr = preprocess_image(abs_path, src)
        if arr is None:
            stats['errors'] += 1
            continue

        save_to_cache(raw, arr)
        stats['processed'] += 1

    return stats


# =========================================================
# PREPROCESSING COMPARISON VISUALIZATION
# =========================================================

def make_preprocessing_comparison(
        save_path: str = None,
        odir_raw_path: str = None,
        aptos_raw_path: str = None) -> str:
    """
    Generate and save a side-by-side comparison PNG showing
    ODIR (CLAHE) vs APTOS (Ben Graham) preprocessing pipelines.

    Returns the saved PNG path.
    """
    if save_path is None:
        save_path = os.path.join(DATA_DIR, 'preprocessing_comparison_v3.png')

    # --- Pick sample ODIR image ---
    # Prefer sample from the dataset
    odir_path = None
    if odir_raw_path:
        odir_path = resolve_image_path(odir_raw_path, 'ODIR')
    if odir_path is None or not os.path.exists(odir_path):
        # Use the one available ODIR sample in odir5k folder
        odir_path = ODIR_SAMPLE
    if not os.path.exists(odir_path):
        # Fall back to any image in odir/preprocessed_images
        imgs = [os.path.join(ODIR_IMG_DIR, f)
                for f in os.listdir(ODIR_IMG_DIR)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        odir_path = imgs[0] if imgs else None

    # --- Pick sample APTOS image ---
    aptos_path = None
    if aptos_raw_path:
        aptos_path = resolve_image_path(aptos_raw_path, 'APTOS')
    if aptos_path is None or not os.path.exists(aptos_path):
        # Use first entry in APTOS lookup
        if _APTOS_LOOKUP:
            aptos_path = next(iter(_APTOS_LOOKUP.values()))

    # --- Load images ---
    def get_or_synthetic(path, name):
        if path and os.path.exists(path):
            img = _load_image(path)
            if img is not None:
                return img, path
        print(f'[WARN] {name} sample not found, using synthetic.')
        h, w = 512, 512
        np.random.seed(42)
        base = np.zeros((h, w, 3), dtype=np.uint8)
        cx, cy = w // 2, h // 2
        r = int(min(h, w) * 0.48)
        cv2.circle(base, (cx, cy), r, (60, 40, 25), -1)
        for _ in range(30):
            pt1 = (cx + np.random.randint(-r, r), cy + np.random.randint(-r, r))
            pt2 = (cx + np.random.randint(-r, r), cy + np.random.randint(-r, r))
            cv2.line(base, pt1, pt2, (100, 60, 35), 1)
        base = base.astype(np.float32) + np.random.normal(0, 6, base.shape)
        return np.clip(base, 0, 255).astype(np.uint8), '(synthetic)'

    odir_orig,  odir_src  = get_or_synthetic(odir_path,  'ODIR')
    aptos_orig, aptos_src = get_or_synthetic(aptos_path, 'APTOS')

    # Resize originals for display
    odir_disp  = cv2.resize(odir_orig,  (TARGET_SIZE, TARGET_SIZE),
                            interpolation=cv2.INTER_AREA)
    aptos_disp = cv2.resize(aptos_orig, (TARGET_SIZE, TARGET_SIZE),
                            interpolation=cv2.INTER_AREA)

    # Apply pipelines
    odir_clahe   = clahe_preprocess(odir_orig.copy())
    aptos_graham = ben_graham_preprocess(aptos_orig.copy())

    # Difference images (scaled for visibility)
    diff_odir  = cv2.absdiff(odir_disp,  odir_clahe)
    diff_aptos = cv2.absdiff(aptos_disp, aptos_graham)
    # Amplify diff for visibility
    diff_odir  = np.clip(diff_odir  * 3, 0, 255).astype(np.uint8)
    diff_aptos = np.clip(diff_aptos * 3, 0, 255).astype(np.uint8)

    # --- Build figure ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle(
        'RetinaSense v3 — Domain-Conditional Preprocessing\n'
        'ODIR: CLAHE Pipeline  |  APTOS: Ben Graham Pipeline',
        fontsize=13, fontweight='bold', color='white', y=1.01
    )

    panels = [
        # row, col, image, title, bg_color
        (0, 0, odir_disp,     f'ODIR: Original\n({os.path.basename(str(odir_src))})',
         '#1565C0'),
        (0, 1, odir_clahe,    'ODIR: After CLAHE\n(L-channel equalization, circular mask)',
         '#0D47A1'),
        (0, 2, diff_odir,     'ODIR: Difference x3\n(|original - CLAHE|, amplified)',
         '#263238'),
        (1, 0, aptos_disp,    f'APTOS: Original\n({os.path.basename(str(aptos_src))})',
         '#BF360C'),
        (1, 1, aptos_graham,  'APTOS: After Ben Graham\n(4*img - 4*blur(σ=10) + 128)',
         '#870000'),
        (1, 2, diff_aptos,    'APTOS: Difference x3\n(|original - Ben Graham|, amplified)',
         '#1B5E20'),
    ]

    for r, c, img_arr, title, fc in panels:
        ax = axes[r, c]
        ax.imshow(img_arr)
        ax.set_title(title, fontsize=9, color='white', pad=5,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor=fc,
                               alpha=0.85, edgecolor='none'))
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Annotation boxes
    odir_note = (
        'ODIR Pipeline\n'
        '━━━━━━━━━━━━━━━\n'
        '1. Crop black borders\n'
        '2. Resize → 224×224\n'
        '3. Convert RGB→LAB\n'
        '4. CLAHE on L channel\n'
        '   clip=2.0, tile=8×8\n'
        '5. LAB→RGB\n'
        '6. Circular mask (r=0.48)'
    )
    aptos_note = (
        'APTOS Pipeline (Ben Graham)\n'
        '━━━━━━━━━━━━━━━━━━━━━━━━━━\n'
        '1. Crop black borders\n'
        '2. Resize → 224×224\n'
        '3. blur = GaussianBlur(σ=10)\n'
        '4. out = 4×img − 4×blur + 128\n'
        '5. Circular mask (r=0.48)\n'
        '6. clip to [0, 255]'
    )

    fig.text(0.02, 0.92, odir_note,  fontsize=8.5, va='top', ha='left',
             color='white', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#1565C0', alpha=0.6))
    fig.text(0.02, 0.48, aptos_note, fontsize=8.5, va='top', ha='left',
             color='white', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#870000', alpha=0.6))

    plt.tight_layout(rect=[0.18, 0, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    plt.close()
    print(f'[OK] Comparison saved: {save_path}')
    return save_path


# =========================================================
# NORMALIZATION STATISTICS
# =========================================================

def compute_norm_stats(train_df: pd.DataFrame,
                       out_path: str = None,
                       max_images: int = None) -> dict:
    """
    Compute per-channel mean and std across all pixels of training images
    after domain-conditional preprocessing. Training set ONLY — no
    validation/test data contamination.

    Returns dict with: mean_rgb, std_rgb, n_images, n_pixels_per_channel.
    """
    if out_path is None:
        out_path = os.path.join(DATA_DIR, 'fundus_norm_stats.json')

    df = train_df.copy()
    if max_images is not None:
        df = df.sample(min(max_images, len(df)), random_state=42)

    ch_sum    = np.zeros(3, dtype=np.float64)
    ch_sq_sum = np.zeros(3, dtype=np.float64)
    n_pixels  = 0
    n_images  = 0
    n_missing = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Norm stats'):
        raw = row['image_path']
        src = row['dataset']

        # Try cache first for speed
        arr = load_from_cache(raw)
        if arr is None:
            abs_path = resolve_image_path(raw, src)
            if not os.path.exists(abs_path):
                n_missing += 1
                continue
            arr = preprocess_image(abs_path, src)
            if arr is None:
                n_missing += 1
                continue

        arr_f   = arr.astype(np.float64) / 255.0
        pixels  = arr_f.reshape(-1, 3)
        ch_sum    += pixels.sum(axis=0)
        ch_sq_sum += (pixels ** 2).sum(axis=0)
        n_pixels  += pixels.shape[0]
        n_images  += 1

    if n_images == 0:
        print('[WARN] No images found — storing ImageNet defaults as fallback.')
        stats = {
            'mean_rgb': [0.485, 0.456, 0.406],
            'std_rgb':  [0.229, 0.224, 0.225],
            'n_images': 0,
            'n_pixels_per_channel': 0,
            'n_missing': n_missing,
            'note': 'No images found — ImageNet defaults used as fallback',
            'source': 'imagenet_fallback'
        }
    else:
        mean = ch_sum    / n_pixels
        var  = ch_sq_sum / n_pixels - mean ** 2
        std  = np.sqrt(np.maximum(var, 0.0))
        stats = {
            'mean_rgb': [round(float(v), 6) for v in mean],
            'std_rgb':  [round(float(v), 6) for v in std],
            'n_images': n_images,
            'n_pixels_per_channel': int(n_pixels),
            'n_missing': n_missing,
            'note': ('Computed on training split only after domain-conditional '
                     'preprocessing. Red-dominant channel expected (fundus tissue).'),
            'source': 'computed_training_split'
        }
        print(f'  mean RGB : {[round(v,4) for v in mean]}')
        print(f'  std  RGB : {[round(v,4) for v in std]}')
        print(f'  images   : {n_images:,}  |  missing: {n_missing}')

    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'[OK] Stats saved: {out_path}')
    return stats


# =========================================================
# 3-WAY STRATIFIED SPLIT
# =========================================================

def create_stratified_split(df: pd.DataFrame,
                             train_ratio: float = 0.70,
                             calib_ratio: float = 0.15,
                             test_ratio:  float = 0.15,
                             random_state: int = 42) -> tuple:
    """
    Create train/calib/test split stratified by disease_label.
    Returns (train_df, calib_df, test_df).
    """
    from sklearn.model_selection import train_test_split as _tts
    assert abs(train_ratio + calib_ratio + test_ratio - 1.0) < 1e-9

    train_df, temp_df = _tts(
        df, test_size=(calib_ratio + test_ratio),
        stratify=df['disease_label'], random_state=random_state
    )
    calib_frac = calib_ratio / (calib_ratio + test_ratio)
    calib_df, test_df = _tts(
        temp_df, test_size=(1.0 - calib_frac),
        stratify=temp_df['disease_label'], random_state=random_state
    )
    return (train_df.reset_index(drop=True),
            calib_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


def save_splits(train_df, calib_df, test_df, out_dir: str = DATA_DIR):
    train_df.to_csv(os.path.join(out_dir, 'train_split.csv'), index=False)
    calib_df.to_csv(os.path.join(out_dir, 'calib_split.csv'), index=False)
    test_df.to_csv( os.path.join(out_dir, 'test_split.csv'),  index=False)
    print(f'[OK] Split CSVs saved to {out_dir}/')


def print_split_stats(train_df, calib_df, test_df,
                      class_names: dict = None) -> str:
    if class_names is None:
        class_names = {0: 'Normal', 1: 'Diabetes/DR', 2: 'Glaucoma',
                       3: 'Cataract', 4: 'AMD'}

    total_n = len(train_df) + len(calib_df) + len(test_df)
    lines = [
        '',
        '=' * 62,
        '  STRATIFIED SPLIT — CLASS DISTRIBUTION',
        '=' * 62,
        f"{'Class':<16} {'Train':>8} {'Calib':>8} {'Test':>8} {'Total':>8}",
        '-' * 54,
    ]
    tr_tot = ca_tot = te_tot = 0
    for lbl in sorted(class_names.keys()):
        tr = int((train_df['disease_label'] == lbl).sum())
        ca = int((calib_df['disease_label'] == lbl).sum())
        te = int((test_df['disease_label']  == lbl).sum())
        tot = tr + ca + te
        tr_tot += tr; ca_tot += ca; te_tot += te
        lines.append(
            f"{class_names[lbl]:<16} {tr:>8,} {ca:>8,} {te:>8,} {tot:>8,}"
        )
    lines += [
        '-' * 54,
        f"{'TOTAL':<16} {tr_tot:>8,} {ca_tot:>8,} {te_tot:>8,} {total_n:>8,}",
        '',
        f'Split sizes : train={len(train_df):,}  calib={len(calib_df):,}  '
        f'test={len(test_df):,}',
        f'Actual ratios: train={len(train_df)/total_n:.1%}  '
        f'calib={len(calib_df)/total_n:.1%}  '
        f'test={len(test_df)/total_n:.1%}',
    ]
    report = '\n'.join(lines)
    print(report)
    return report


# =========================================================
# ADDITIONAL DATASET SEARCH
# =========================================================

def search_additional_datasets() -> dict:
    """
    Scan filesystem for REFUGE2, iChallenge-AMD, RIM-ONE and other
    AMD/Glaucoma-specific datasets beyond the current CSV.
    Returns a findings dict.
    """
    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    TARGETS  = ['refuge2', 'refuge', 'ichallenge', 'rim-one', 'rimone',
                'amd', 'glaucoma', 'odir5k', 'odir']
    SEARCH_ROOTS = ['/teamspace/studios/this_studio', '/teamspace/uploads']
    SKIP_DIRS    = {'.git', '.cache', '.claude', '.ipython', '.npm',
                    '__pycache__', 'outputs_analysis', 'outputs_ensemble',
                    'outputs_optimized', 'outputs_production', 'outputs_v2',
                    'outputs_v2_extended', 'outputs_vit'}

    findings = {}

    for root_dir in SEARCH_ROOTS:
        if not os.path.exists(root_dir):
            continue
        for dirpath, dirnames, files in os.walk(root_dir):
            # Prune
            dirnames[:] = [d for d in dirnames
                           if d not in SKIP_DIRS and not d.startswith('.')]
            folder = os.path.basename(dirpath).lower()
            for target in TARGETS:
                if target in folder:
                    img_cnt = sum(1 for f in files
                                  if os.path.splitext(f)[1].lower() in IMG_EXTS)
                    key = dirpath
                    if key not in findings or img_cnt > findings[key]['img_count']:
                        findings[key] = {
                            'matched_target': target,
                            'img_count': img_cnt,
                            'total_files': len(files)
                        }

    # Always include the known special dirs
    for special in [
        '/teamspace/studios/this_studio/ocular-disease-recognition-odir5k',
        '/teamspace/studios/this_studio/odir',
        '/teamspace/studios/this_studio/aptos',
    ]:
        if os.path.exists(special) and special not in findings:
            img_cnt = sum(
                1 for root, _, files in os.walk(special)
                for f in files
                if os.path.splitext(f)[1].lower() in IMG_EXTS
            )
            findings[special] = {
                'matched_target': 'known_dataset',
                'img_count': img_cnt,
                'total_files': sum(1 for _, _, fs in os.walk(special) for _ in fs)
            }

    return findings


# =========================================================
# MAIN
# =========================================================

def main():
    print('=' * 65)
    print('  RetinaSense v3 — Data Pipeline')
    print('=' * 65)

    CLASS_NAMES = {0: 'Normal', 1: 'Diabetes/DR', 2: 'Glaucoma',
                   3: 'Cataract', 4: 'AMD'}

    # -------------------------------------------------------
    # TASK 1: Dataset Audit
    # -------------------------------------------------------
    print('\n[TASK 1] Dataset Audit')
    print('-' * 50)
    df = pd.read_csv(CSV_PATH)
    print(f'  CSV          : {CSV_PATH}')
    print(f'  Total rows   : {len(df):,}')
    print(f'  Columns      : {df.columns.tolist()}')
    print()

    print('  --- Overall class distribution ---')
    for lbl, cnt in df['disease_label'].value_counts().sort_index().items():
        pct = cnt / len(df) * 100
        bar = '#' * int(pct / 2)
        print(f"    {lbl} {CLASS_NAMES.get(lbl,'?'):<12} : {cnt:>5}  ({pct:5.1f}%)  {bar}")

    max_cls = df['disease_label'].value_counts().max()
    min_cls = df['disease_label'].value_counts().min()
    print(f'\n  Imbalance ratio (max/min): {max_cls/min_cls:.1f}:1')
    print()

    print('  --- Per-dataset breakdown ---')
    per_ds = (df.groupby(['dataset', 'disease_label'])
               .size().reset_index(name='count'))
    print(per_ds.to_string(index=False))
    print()

    print('  --- Severity label distribution (APTOS only) ---')
    for sev, cnt in df['severity_label'].value_counts().sort_index().items():
        label = 'N/A (ODIR)' if sev == -1 else f'Grade {sev}'
        print(f"    {sev:>3} ({label:<14}): {cnt:>5}")
    print()

    print('  --- Image path existence check ---')
    n_found = 0
    for _, row in df.iterrows():
        p = resolve_image_path(row['image_path'], row['dataset'])
        if os.path.exists(p):
            n_found += 1
    n_missing = len(df) - n_found
    print(f'    Total checked : {len(df):,}')
    print(f'    Found on disk : {n_found:,}')
    print(f'    Missing       : {n_missing:,}')
    print()

    # -------------------------------------------------------
    # TASK 2: Preprocessing Comparison
    # -------------------------------------------------------
    print('[TASK 2] Domain-Conditional Preprocessing Comparison')
    print('-' * 50)

    # Get representative samples from each dataset
    odir_sample  = df[df['dataset'] == 'ODIR']['image_path'].iloc[0] \
                   if len(df[df['dataset'] == 'ODIR']) > 0 else None
    aptos_sample = df[df['dataset'] == 'APTOS']['image_path'].iloc[0] \
                   if len(df[df['dataset'] == 'APTOS']) > 0 else None

    comp_path = make_preprocessing_comparison(
        odir_raw_path=odir_sample,
        aptos_raw_path=aptos_sample
    )

    # Demo: process a few images to verify pipeline
    print('\n  --- Pipeline verification (5 ODIR + 5 APTOS) ---')
    ok_odir = ok_aptos = 0
    for _, row in df[df['dataset'] == 'ODIR'].head(5).iterrows():
        p = resolve_image_path(row['image_path'], 'ODIR')
        if os.path.exists(p):
            arr = preprocess_image(p, 'ODIR')
            if arr is not None and arr.shape == (TARGET_SIZE, TARGET_SIZE, 3):
                ok_odir += 1
    for _, row in df[df['dataset'] == 'APTOS'].head(5).iterrows():
        p = resolve_image_path(row['image_path'], 'APTOS')
        if os.path.exists(p):
            arr = preprocess_image(p, 'APTOS')
            if arr is not None and arr.shape == (TARGET_SIZE, TARGET_SIZE, 3):
                ok_aptos += 1
    print(f'    ODIR  (CLAHE)      : {ok_odir}/5 OK')
    print(f'    APTOS (Ben Graham) : {ok_aptos}/5 OK')
    print()

    # -------------------------------------------------------
    # TASK 3: Stratified Split
    # -------------------------------------------------------
    print('[TASK 3] 3-Way Stratified Split (70 / 15 / 15)')
    print('-' * 50)
    train_df, calib_df, test_df = create_stratified_split(df)
    save_splits(train_df, calib_df, test_df)
    split_report = print_split_stats(train_df, calib_df, test_df, CLASS_NAMES)
    print()

    # -------------------------------------------------------
    # TASK 4: Normalization Statistics (training split only)
    # -------------------------------------------------------
    print('[TASK 4] Fundus Normalization Statistics (training split)')
    print('-' * 50)
    norm_stats = compute_norm_stats(train_df)
    print()

    # -------------------------------------------------------
    # TASK 5: Additional Dataset Search
    # -------------------------------------------------------
    print('[TASK 5] Additional Dataset Search')
    print('-' * 50)
    findings = search_additional_datasets()
    if findings:
        print(f'  Found {len(findings)} dataset directories:')
        for path, info in findings.items():
            print(f'    {path}')
            print(f'      images: {info["img_count"]:,}  '
                  f'files: {info["total_files"]:,}  '
                  f'matched: "{info["matched_target"]}"')
    else:
        print('  No additional datasets found.')
    print()

    # Summary of what needs downloading
    known_sets = {'REFUGE2', 'ICHALLENGE-AMD', 'RIM-ONE'}
    found_names = set(info['matched_target'].upper()
                      for info in findings.values())
    missing_sets = known_sets - found_names
    if missing_sets:
        print(f'  Datasets NOT found (need downloading): {missing_sets}')

    # -------------------------------------------------------
    # Write report
    # -------------------------------------------------------
    _write_report(df, train_df, calib_df, test_df, norm_stats,
                  findings, split_report, comp_path)

    print('\n' + '=' * 65)
    print('  All tasks complete.')
    print('=' * 65)
    return df, train_df, calib_df, test_df, norm_stats


# =========================================================
# REPORT WRITER
# =========================================================

def _write_report(df, train_df, calib_df, test_df, norm_stats,
                  dataset_findings, split_report, comp_path):
    """Save data_engineer_report.md to ./data/"""
    CLASS_NAMES = {0: 'Normal', 1: 'Diabetes/DR', 2: 'Glaucoma',
                   3: 'Cataract', 4: 'AMD'}

    n_found = sum(
        1 for _, row in df.iterrows()
        if os.path.exists(resolve_image_path(row['image_path'], row['dataset']))
    )

    lines = [
        '# RetinaSense v3 — Data Engineer Report',
        f'Generated: 2026-03-06',
        '',
        '---',
        '',
        '## 1. Dataset Statistics',
        '',
        f'**Source CSV:** `data/combined_dataset.csv`  ',
        f'**Total images in CSV:** {len(df):,}  ',
        f'**Images found on disk:** {n_found:,} / {len(df):,}  ',
        '',
        '### Source breakdown',
        '',
        '| Dataset | Count | Labels present |',
        '|---------|-------|----------------|',
    ]
    for ds, grp in df.groupby('dataset'):
        labels = sorted(grp['disease_label'].unique())
        label_str = ', '.join(f'{l}={CLASS_NAMES[l]}' for l in labels)
        lines.append(f'| {ds} | {len(grp):,} | {label_str} |')

    lines += [
        '',
        '### Class distribution (full dataset)',
        '',
        '| Label | Class | Count | % |',
        '|-------|-------|-------|---|',
    ]
    for lbl, cnt in df['disease_label'].value_counts().sort_index().items():
        pct = cnt / len(df) * 100
        lines.append(
            f'| {lbl} | {CLASS_NAMES[lbl]} | {cnt:,} | {pct:.1f}% |'
        )
    max_cls = df['disease_label'].value_counts().max()
    min_cls = df['disease_label'].value_counts().min()
    lines += [
        '',
        f'**Imbalance ratio (Diabetes:AMD):** {max_cls/min_cls:.1f}:1',
        '',
        '### Severity label distribution (APTOS DR grades, -1 = ODIR no grade)',
        '',
        '| Severity | Meaning | Count |',
        '|----------|---------|-------|',
    ]
    for sev, cnt in df['severity_label'].value_counts().sort_index().items():
        meaning = 'N/A (ODIR, no grade)' if sev == -1 else f'DR Grade {sev}'
        lines.append(f'| {sev} | {meaning} | {cnt:,} |')

    lines += [
        '',
        '---',
        '',
        '## 2. Image Path Resolution',
        '',
        '| Dataset | CSV path format | Actual location |',
        '|---------|-----------------|-----------------|',
        '| ODIR  | `.//odir/preprocessed_images/<name>.jpg` | `odir/preprocessed_images/<name>.jpg` |',
        '| APTOS | `.//aptos/train_images/<id>.png` (train_images does NOT exist) | `aptos/gaussian_filtered_images/gaussian_filtered_images/<class>/<id>.png` |',
        '',
        '`train_images/` directory is absent; actual APTOS images are stored under',
        '`gaussian_filtered_images/gaussian_filtered_images/<DR_grade>/`. The',
        '`aptos/train.csv` maps `id_code` → `diagnosis` (0-4) enabling lookup.',
        '',
        '---',
        '',
        '## 3. Preprocessing: Domain-Conditional Pipeline',
        '',
        '**Problem:** Previous versions applied Ben Graham enhancement uniformly to',
        'ALL images. This is incorrect: ODIR images have already-enhanced or',
        'clinical-quality appearance; applying Ben Graham degrades them.',
        '',
        '**Fix:** Source-conditional dispatch in `preprocess_image(path, source)`.',
        '',
        '| Source | Method | Rationale |',
        '|--------|--------|-----------|',
        '| APTOS  | Ben Graham (4×img − 4×blur(σ=10) + 128 + circular mask) | Field camera images have vignetting and low local contrast. Ben Graham removes low-frequency illumination and amplifies vessel/lesion detail. |',
        '| ODIR   | CLAHE (L-channel, clip=2.0, tile=8×8, circular mask) | Multi-source clinical images. CLAHE normalizes local contrast while preserving sharpness and avoiding Ben Graham over-processing. |',
        '| REFUGE2 | Resize only (224×224) | Zeiss Visucam 500 — already standardized high-quality. |',
        '',
        f'**Comparison figure:** `{comp_path}`',
        '',
        '**Cache location:** `preprocessed_cache_v3/<stem>_v3.npy`  ',
        '**Cache key:** image filename stem (not row index)',
        '',
        '---',
        '',
        '## 4. Normalization Statistics',
        '',
        '**Method:** One pass over training split pixels (post-preprocessing).',
        'No validation or test images used.',
        '',
        f'| Channel | Mean | Std |',
        f'|---------|------|-----|',
        f'| R (red)   | {norm_stats["mean_rgb"][0]:.4f} | {norm_stats["std_rgb"][0]:.4f} |',
        f'| G (green) | {norm_stats["mean_rgb"][1]:.4f} | {norm_stats["std_rgb"][1]:.4f} |',
        f'| B (blue)  | {norm_stats["mean_rgb"][2]:.4f} | {norm_stats["std_rgb"][2]:.4f} |',
        '',
        f'**Images used:** {norm_stats["n_images"]:,}  ',
        f'**Note:** {norm_stats["note"]}  ',
        f'**Source:** `{norm_stats["source"]}`',
    ]

    if norm_stats['source'] == 'computed_training_split':
        lines += [
            '',
            'Expected pattern for fundus images: R > G > B (red-dominant)',
            'due to high hemoglobin absorption. Computed values should match',
            'expected ≈ [0.41, 0.27, 0.19] mean, [0.28, 0.19, 0.16] std.',
        ]

    lines += [
        '',
        '**Saved to:** `data/fundus_norm_stats.json`',
        '',
        '---',
        '',
        '## 5. Stratified Split (70 / 15 / 15)',
        '',
        '**Strategy:** `sklearn.model_selection.train_test_split` with',
        '`stratify=disease_label`, `random_state=42`.',
        '',
        '**Files:**',
        '- `data/train_split.csv` — 70% training',
        '- `data/calib_split.csv` — 15% calibration (temperature scaling)',
        '- `data/test_split.csv`  — 15% held-out evaluation',
        '',
    ]
    lines.append(split_report.replace('\n', '\n'))
    lines += [
        '',
        '---',
        '',
        '## 6. Additional Dataset Search',
        '',
    ]
    if dataset_findings:
        lines.append('### Found directories:')
        lines.append('')
        lines.append('| Path | Images | Files | Matched |')
        lines.append('|------|--------|-------|---------|')
        for path, info in dataset_findings.items():
            lines.append(
                f'| `{path}` | {info["img_count"]:,} | '
                f'{info["total_files"]:,} | {info["matched_target"]} |'
            )
    else:
        lines.append('No additional dataset directories found.')

    lines += [
        '',
        '### Availability summary',
        '',
        '| Dataset | Status | Location |',
        '|---------|--------|----------|',
        '| ODIR-5K (ODIR) | **AVAILABLE** | `odir/preprocessed_images/` (4,878 images in CSV) |',
        '| ODIR-5K raw    | **AVAILABLE** | `odir/ODIR-5K/ODIR-5K/Training Images/` (7,000) + Testing (1,000) |',
        '| APTOS 2019     | **AVAILABLE** | `aptos/gaussian_filtered_images/` (3,662 images) |',
        '| ocular-disease-recognition-odir5k | Partial (1 image only) | `ocular-disease-recognition-odir5k/preprocessed_images/` |',
        '| REFUGE2        | **NOT FOUND** | Needs download |',
        '| iChallenge-AMD | **NOT FOUND** | Needs download |',
        '| RIM-ONE        | **NOT FOUND** | Needs download |',
        '',
        '### AMD / Glaucoma specific images (beyond CSV)',
        '',
        f'- ODIR provides {len(df[df["disease_label"]==2]):,} Glaucoma and '
        f'{len(df[df["disease_label"]==4]):,} AMD images from '
        f'`odir/preprocessed_images/`.',
        '- ODIR raw training set (7,000 images) may contain additional',
        '  AMD/Glaucoma cases not yet extracted — check `odir/full_df.csv`.',
        '- For specialized Glaucoma detection: REFUGE2 (400 images,',
        '  Magrabia population) and RIM-ONE (159 images) are recommended.',
        '- For AMD: iChallenge-AMD (400 images) is the standard benchmark.',
        '',
        '---',
        '',
        '## 7. Action Items',
        '',
        '1. **Download missing datasets** to improve minority class coverage:',
        '   - REFUGE2: https://refuge.grand-challenge.org/',
        '   - RIM-ONE: http://medimrg.webs.ull.es/research/retinal-imaging/rim-one/',
        '   - iChallenge-AMD: https://amd.grand-challenge.org/',
        '2. **Fix paths in combined_dataset.csv**: update `aptos/train_images/` →',
        '   actual `gaussian_filtered_images/.../` paths.',
        '3. **Run full cache build** when training: `python retinasense_v3_preprocessing.py --cache-all`',
        '4. **Use computed normalization stats** from `data/fundus_norm_stats.json`',
        '   instead of ImageNet stats.',
        '5. **Address 21:1 class imbalance**: consider weighted sampling or',
        '   oversampling minority classes (AMD=265, Glaucoma=308).',
    ]

    report_path = os.path.join(DATA_DIR, 'data_engineer_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'[OK] Report saved: {report_path}')


if __name__ == '__main__':
    main()
