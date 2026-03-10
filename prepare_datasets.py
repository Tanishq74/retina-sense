#!/usr/bin/env python3
"""
RetinaSense — Multi-Dataset Preparation Pipeline
==================================================
Standardises, preprocesses, and merges public retinal fundus datasets into
a unified training corpus.  Designed to expand the current 8,540-image
(APTOS + ODIR) dataset with additional glaucoma, AMD, and DR sources.

Supported datasets
------------------
  EyePACS      ~35 K images, DR severity 0-4 (Kaggle)
  MESSIDOR-2    1,748 images, DR grades       (ADCIS)
  REFUGE/REFUGE2 ~1,200 images, glaucoma      (Grand Challenge)
  ADAM (iChallenge-AMD) ~1,200 images, AMD    (Grand Challenge)
  ORIGA         ~650 images, glaucoma          (SiMES study)

Key design decision — UNIFIED CLAHE preprocessing
--------------------------------------------------
The existing v3 pipeline applies *different* enhancement per source
(Ben Graham for APTOS, CLAHE for ODIR, resize-only for REFUGE2).
When many heterogeneous sources are mixed, a single consistent pipeline
avoids distribution shifts caused by preprocessing discrepancies.
All new datasets go through:

    crop borders -> resize 224x224 -> CLAHE (L-channel) -> circular mask

This matches the ODIR branch of the v3 pipeline and is the safest
default for unknown-quality fundus images.

Output CSV columns  (matches existing format)
----------------------------------------------
  image_path, disease_label, source, severity_label, cache_path

Usage examples
--------------
  # Prepare a single dataset
  python prepare_datasets.py --dataset eyepacs --raw-dir ./data/eyepacs --output-dir ./data/

  # Prepare all downloaded datasets
  python prepare_datasets.py --all --output-dir ./data/

  # Print download instructions
  python prepare_datasets.py --dataset refuge --instructions

  # Merge all prepared CSVs into unified corpus with splits
  python prepare_datasets.py --merge --output-dir ./data/

CPU-only — no GPU required.
"""

import os
import sys
import hashlib
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =====================================================================
# CONSTANTS
# =====================================================================

TARGET_SIZE = 224
CLASS_NAMES = {0: "Normal", 1: "Diabetes/DR", 2: "Glaucoma", 3: "Cataract", 4: "AMD"}
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# =====================================================================
# UNIFIED PREPROCESSING  (single pipeline for ALL sources)
# =====================================================================


def _load_image(image_path: str) -> Optional[np.ndarray]:
    """Load image as RGB numpy array (H, W, 3) uint8.  Returns None on failure."""
    img = cv2.imread(image_path)
    if img is None:
        try:
            from PIL import Image as PILImage
            pil = PILImage.open(image_path).convert("RGB")
            img = np.array(pil)
            return img  # already RGB
        except Exception:
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
    return img[rmin : rmax + 1, cmin : cmax + 1]


def _apply_circular_mask(img: np.ndarray) -> np.ndarray:
    """Zero out pixels outside the circular fundus field of view."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    r = int(min(h, w) * 0.48)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    return cv2.bitwise_and(img, img, mask=mask)


def unified_preprocess(
    img_path: str,
    target_size: int = TARGET_SIZE,
    clip_limit: float = 2.0,
    tile_grid: Tuple[int, int] = (8, 8),
) -> Optional[np.ndarray]:
    """
    Unified CLAHE preprocessing pipeline for ALL dataset sources.

    Steps:
        1. Load image as RGB
        2. Crop black borders (dark-pixel trimming, tolerance=7)
        3. Resize to target_size x target_size (INTER_AREA for downscaling quality)
        4. Convert RGB -> LAB colour space
        5. Apply CLAHE to L (luminance) channel only
           - clip_limit=2.0 prevents noise amplification
           - tile_grid=(8,8) gives local adaptation at appropriate spatial scale
        6. Convert LAB -> RGB
        7. Apply circular mask (radius=0.48*side, zeros outside)
        8. Clip to [0, 255], return as uint8

    Parameters
    ----------
    img_path : str
        Absolute path to the fundus image file.
    target_size : int
        Output spatial dimension (square).  Default 224.
    clip_limit : float
        CLAHE clipping limit.
    tile_grid : tuple
        CLAHE tile grid size.

    Returns
    -------
    np.ndarray of shape (target_size, target_size, 3), dtype uint8,
    or None if the image cannot be loaded.
    """
    img = _load_image(img_path)
    if img is None:
        return None

    img = _crop_black_borders(img)
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # CLAHE on luminance channel only (preserves hue)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l_ch)
    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    img = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    img = _apply_circular_mask(img)
    return np.clip(img, 0, 255).astype(np.uint8)


# =====================================================================
# DATASET CONFIGS
# =====================================================================


@dataclass
class DatasetConfig:
    """Configuration for one public retinal dataset."""

    name: str
    description: str
    approximate_size: str
    url: str
    label_type: str  # e.g. "DR_severity", "glaucoma_binary", "amd_binary"
    disease_label_map: Dict  # maps raw label -> RetinaSense disease_label
    download_commands: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# -- EyePACS (Kaggle Diabetic Retinopathy Detection) ------------------

EYEPACS_CONFIG = DatasetConfig(
    name="eyepacs",
    description="Kaggle Diabetic Retinopathy Detection (EyePACS)",
    approximate_size="~35,126 train images + 53,576 test images",
    url="https://www.kaggle.com/c/diabetic-retinopathy-detection/data",
    label_type="DR_severity",
    disease_label_map={
        0: 0,  # No DR          -> Normal
        1: 1,  # Mild NPDR      -> Diabetes/DR
        2: 1,  # Moderate NPDR  -> Diabetes/DR
        3: 1,  # Severe NPDR    -> Diabetes/DR
        4: 1,  # PDR            -> Diabetes/DR
    },
    download_commands=[
        "# Requires Kaggle API credentials (~/.kaggle/kaggle.json)",
        "pip install kaggle",
        "kaggle competitions download -c diabetic-retinopathy-detection",
        "unzip diabetic-retinopathy-detection.zip -d ./data/eyepacs/",
        "# Expected structure:",
        "#   data/eyepacs/train/       (JPEG images)",
        "#   data/eyepacs/trainLabels.csv",
    ],
    notes=[
        "Images are high-resolution JPEG from EyePACS screening cameras.",
        "trainLabels.csv has columns: image, level (0-4 DR severity).",
        "Severity 0 maps to Normal (disease_label=0), severity 1-4 to DR (disease_label=1).",
        "Many images have significant black borders that need cropping.",
    ],
)

# -- MESSIDOR-2 --------------------------------------------------------

MESSIDOR2_CONFIG = DatasetConfig(
    name="messidor2",
    description="MESSIDOR-2 (Methods to Evaluate Segmentation and Indexing Techniques)",
    approximate_size="~1,748 images",
    url="https://www.adcis.net/en/third-party/messidor2/",
    label_type="DR_severity",
    disease_label_map={
        0: 0,  # No DR          -> Normal
        1: 1,  # Mild NPDR      -> Diabetes/DR
        2: 1,  # Moderate NPDR  -> Diabetes/DR
        3: 1,  # Severe NPDR    -> Diabetes/DR
    },
    download_commands=[
        "# Request access at: https://www.adcis.net/en/third-party/messidor2/",
        "# OR use the Kaggle mirror:",
        "kaggle datasets download -d google-brain/messidor2-dr-grades",
        "unzip messidor2-dr-grades.zip -d ./data/messidor2/",
        "# Expected structure:",
        "#   data/messidor2/images/            (TIFF or JPEG images)",
        "#   data/messidor2/messidor_data.csv  (or similar labels CSV)",
    ],
    notes=[
        "Original images are 1440x960 or 2240x1488 TIFF.",
        "Label CSV typically has columns: image_id, adjudicated_dr_grade.",
        "Grades 0-3 (R0=no DR, R1=mild, R2=moderate, R3=severe).",
        "Grade 0 -> Normal, grades 1-3 -> DR.",
    ],
)

# -- REFUGE / REFUGE2 --------------------------------------------------

REFUGE_CONFIG = DatasetConfig(
    name="refuge",
    description="REFUGE / REFUGE2 (Retinal Fundus Glaucoma Challenge)",
    approximate_size="~1,200 images (400 train + 400 val + 400 test)",
    url="https://refuge.grand-challenge.org/",
    label_type="glaucoma_binary",
    disease_label_map={
        0: 0,  # Non-glaucoma -> Normal
        1: 2,  # Glaucoma     -> Glaucoma (class 2)
    },
    download_commands=[
        "# Register and download from Grand Challenge:",
        "#   https://refuge.grand-challenge.org/",
        "# OR use the Kaggle mirror:",
        "kaggle datasets download -d andrewmvd/refuge-challenge",
        "unzip refuge-challenge.zip -d ./data/refuge/",
        "# Expected structure:",
        "#   data/refuge/Training400/  (Glaucoma/ and Non-Glaucoma/ subdirs)",
        "#   data/refuge/Validation400/",
        "#   data/refuge/Test400/",
        "#   (each subdir contains .jpg fundus images)",
    ],
    notes=[
        "Images acquired with Zeiss Visucam 500 -- clinical-grade quality.",
        "Directory structure encodes labels: Glaucoma/ vs Non-Glaucoma/.",
        "Training400 has 40 glaucoma + 360 non-glaucoma (10% prevalence).",
        "REFUGE2 is the updated version with additional modalities.",
    ],
)

# -- iChallenge-AMD (ADAM) ---------------------------------------------

ADAM_CONFIG = DatasetConfig(
    name="adam",
    description="ADAM / iChallenge-AMD (Age-related Macular Degeneration)",
    approximate_size="~1,200 images (400 train + 400 val + 400 test)",
    url="https://amd.grand-challenge.org/",
    label_type="amd_binary",
    disease_label_map={
        0: 0,  # Non-AMD -> Normal
        1: 4,  # AMD     -> AMD (class 4)
    },
    download_commands=[
        "# Register and download from Grand Challenge:",
        "#   https://amd.grand-challenge.org/",
        "# OR use the Kaggle mirror:",
        "kaggle datasets download -d andrewmvd/ichallenge-amd",
        "unzip ichallenge-amd.zip -d ./data/adam/",
        "# Expected structure:",
        "#   data/adam/Training400/       (AMD/ and Non-AMD/ subdirs)",
        "#   data/adam/Validation400/",
        "#   data/adam/Test400/",
        "#   OR:",
        "#   data/adam/images/            (flat directory)",
        "#   data/adam/adam_labels.csv     (image_id, label columns)",
    ],
    notes=[
        "Some distributions use directory-based labels (AMD/ vs Non-AMD/).",
        "Others provide a CSV with columns: imgName, label (0 or 1).",
        "This script handles both structures.",
        "AMD prevalence in training is ~25% (100/400).",
    ],
)

# -- ORIGA -------------------------------------------------------------

ORIGA_CONFIG = DatasetConfig(
    name="origa",
    description="ORIGA (Online Retinal fundus Image database for Glaucoma Analysis)",
    approximate_size="~650 images",
    url="https://drive.google.com/drive/folders/1bHnVDptnFmXpwCNJsm3aDRBN0RLaIJ3Y",
    label_type="glaucoma_binary",
    disease_label_map={
        0: 0,  # Non-glaucoma -> Normal
        1: 2,  # Glaucoma     -> Glaucoma (class 2)
    },
    download_commands=[
        "# ORIGA is from the Singapore Malay Eye Study (SiMES).",
        "# Public mirror (subject to availability):",
        "#   https://drive.google.com/drive/folders/1bHnVDpwCNJsm3aDRBN0RLaIJ3Y",
        "# OR search for 'ORIGA glaucoma dataset' on IEEE DataPort / Papers With Code.",
        "#",
        "# Download and extract to:",
        "mkdir -p ./data/origa/images",
        "# Place images in: data/origa/images/",
        "# Place labels in: data/origa/origa_labels.csv",
        "#   (columns: Filename, Glaucoma -- where Glaucoma is 0 or 1)",
    ],
    notes=[
        "650 images from SiMES: 168 glaucoma, 482 non-glaucoma.",
        "Label file may be .xlsx or .csv depending on source.",
        "Images are ~3072x2048 JPEG, need significant downscaling.",
    ],
)

ALL_CONFIGS = {
    "eyepacs": EYEPACS_CONFIG,
    "messidor2": MESSIDOR2_CONFIG,
    "refuge": REFUGE_CONFIG,
    "adam": ADAM_CONFIG,
    "origa": ORIGA_CONFIG,
}


# =====================================================================
# DOWNLOAD INSTRUCTIONS
# =====================================================================


def download_instructions(config: DatasetConfig) -> None:
    """Print download instructions for a dataset."""
    print()
    print("=" * 70)
    print(f"  {config.name.upper()} — Download Instructions")
    print(f"  {config.description}")
    print("=" * 70)
    print()
    print(f"  URL:  {config.url}")
    print(f"  Size: {config.approximate_size}")
    print()
    print("  Commands:")
    for cmd in config.download_commands:
        print(f"    {cmd}")
    print()
    if config.notes:
        print("  Notes:")
        for note in config.notes:
            print(f"    - {note}")
    print()


# =====================================================================
# DATASET-SPECIFIC PREPARE FUNCTIONS
# =====================================================================


def _find_images(directory: str) -> List[str]:
    """Recursively find all image files under a directory."""
    images = []
    for root, _, files in os.walk(directory):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS:
                images.append(os.path.join(root, f))
    return sorted(images)


def _find_csv(directory: str, candidates: List[str]) -> Optional[str]:
    """Find the first existing CSV from a list of candidate filenames."""
    for name in candidates:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            return path
    # Fallback: search for any CSV
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(".csv"):
                return os.path.join(root, f)
    return None


def prepare_eyepacs(raw_dir: str, output_csv: str) -> pd.DataFrame:
    """
    Prepare EyePACS (Kaggle Diabetic Retinopathy Detection).

    Expected structure:
        raw_dir/train/              (or raw_dir/train_images/)
        raw_dir/trainLabels.csv     (columns: image, level)
    """
    config = EYEPACS_CONFIG
    print(f"\n[{config.name.upper()}] Preparing from: {raw_dir}")

    # Find labels CSV
    labels_csv = _find_csv(raw_dir, [
        "trainLabels.csv",
        "trainLabels.csv.zip",
        "train_labels.csv",
        "labels.csv",
    ])
    if labels_csv is None:
        raise FileNotFoundError(
            f"No labels CSV found in {raw_dir}. "
            "Expected trainLabels.csv with columns: image, level"
        )

    labels_df = pd.read_csv(labels_csv)
    print(f"  Labels CSV: {labels_csv} ({len(labels_df)} entries)")

    # Identify the image ID and severity columns
    id_col = None
    for c in ["image", "id_code", "Image", "ID"]:
        if c in labels_df.columns:
            id_col = c
            break
    if id_col is None:
        id_col = labels_df.columns[0]

    level_col = None
    for c in ["level", "diagnosis", "dr_grade", "Grade", "severity"]:
        if c in labels_df.columns:
            level_col = c
            break
    if level_col is None:
        level_col = labels_df.columns[1]

    print(f"  Using columns: id='{id_col}', severity='{level_col}'")

    # Find image directory
    img_dir = None
    for candidate in ["train", "train_images", "images", "train_images_resized"]:
        d = os.path.join(raw_dir, candidate)
        if os.path.isdir(d):
            img_dir = d
            break
    if img_dir is None:
        img_dir = raw_dir  # images may be directly in raw_dir

    print(f"  Image directory: {img_dir}")

    # Build the output dataframe
    rows = []
    found = 0
    missing = 0
    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="  Scanning"):
        img_id = str(row[id_col]).strip()
        severity = int(row[level_col])
        disease_label = config.disease_label_map.get(severity, 1)

        # Try to find the image file
        img_path = None
        for ext in [".jpeg", ".jpg", ".png", ".tif", ""]:
            candidate_path = os.path.join(img_dir, img_id + ext)
            if os.path.exists(candidate_path):
                img_path = os.path.abspath(candidate_path)
                break
        # Also try without extension if file already has one
        if img_path is None:
            full = os.path.join(img_dir, img_id)
            if os.path.exists(full):
                img_path = os.path.abspath(full)

        if img_path is None:
            missing += 1
            continue

        found += 1
        rows.append({
            "image_path": img_path,
            "disease_label": disease_label,
            "source": "EYEPACS",
            "severity_label": severity,
            "cache_path": "",
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"  Found: {found} | Missing: {missing} | Saved: {output_csv}")
    _print_class_dist(df)
    return df


def prepare_messidor2(raw_dir: str, output_csv: str) -> pd.DataFrame:
    """
    Prepare MESSIDOR-2.

    Expected structure:
        raw_dir/images/                  (TIFF/JPEG images)
        raw_dir/messidor_data.csv        (columns: image_id, adjudicated_dr_grade)
    """
    config = MESSIDOR2_CONFIG
    print(f"\n[{config.name.upper()}] Preparing from: {raw_dir}")

    labels_csv = _find_csv(raw_dir, [
        "messidor_data.csv",
        "messidor2_labels.csv",
        "labels.csv",
        "Annotation_Base11.csv",
        "Annotation_Base12.csv",
        "Annotation_Base13.csv",
        "Annotation_Base14.csv",
    ])

    if labels_csv is not None:
        labels_df = pd.read_csv(labels_csv)
        print(f"  Labels CSV: {labels_csv} ({len(labels_df)} entries)")

        # Detect column names
        id_col = None
        for c in ["image_id", "Image name", "image", "ID", "filename"]:
            if c in labels_df.columns:
                id_col = c
                break
        if id_col is None:
            id_col = labels_df.columns[0]

        grade_col = None
        for c in [
            "adjudicated_dr_grade", "Retinopathy grade", "dr_grade",
            "grade", "level", "DR",
        ]:
            if c in labels_df.columns:
                grade_col = c
                break
        if grade_col is None:
            grade_col = labels_df.columns[1]

        print(f"  Using columns: id='{id_col}', grade='{grade_col}'")

        # Find image directory
        img_dir = None
        for candidate in ["images", "IMAGES", "Img", "Base11", "Base12", "Base13", "Base14"]:
            d = os.path.join(raw_dir, candidate)
            if os.path.isdir(d):
                img_dir = d
                break
        if img_dir is None:
            img_dir = raw_dir

        rows = []
        found = 0
        missing = 0
        for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="  Scanning"):
            img_id = str(row[id_col]).strip()
            grade = int(row[grade_col])
            disease_label = config.disease_label_map.get(grade, 1)

            img_path = None
            for ext in [".tif", ".tiff", ".jpg", ".jpeg", ".png", ""]:
                candidate_path = os.path.join(img_dir, img_id + ext)
                if os.path.exists(candidate_path):
                    img_path = os.path.abspath(candidate_path)
                    break
            if img_path is None:
                full = os.path.join(img_dir, img_id)
                if os.path.exists(full):
                    img_path = os.path.abspath(full)

            if img_path is None:
                missing += 1
                continue

            found += 1
            rows.append({
                "image_path": img_path,
                "disease_label": disease_label,
                "source": "MESSIDOR2",
                "severity_label": grade,
                "cache_path": "",
            })

        df = pd.DataFrame(rows)
    else:
        # No CSV -- infer from directory structure or image names
        print("  No labels CSV found. Scanning images directory...")
        all_imgs = _find_images(raw_dir)
        print(f"  Found {len(all_imgs)} images (all labelled as DR=1 without grade info)")
        rows = []
        for img_path in all_imgs:
            rows.append({
                "image_path": os.path.abspath(img_path),
                "disease_label": 1,  # DR assumed without labels
                "source": "MESSIDOR2",
                "severity_label": -1,
                "cache_path": "",
            })
        df = pd.DataFrame(rows)
        print("  WARNING: No grade labels -- all images assigned disease_label=1 (DR).")
        print("  Provide a labels CSV for proper Normal/DR separation.")

    df.to_csv(output_csv, index=False)
    print(f"  Saved: {output_csv}")
    _print_class_dist(df)
    return df


def prepare_refuge(raw_dir: str, output_csv: str) -> pd.DataFrame:
    """
    Prepare REFUGE / REFUGE2.

    Handles two common structures:
      A) Directory-based labels:
           raw_dir/Training400/Glaucoma/     and  .../Non-Glaucoma/
           raw_dir/Validation400/Glaucoma/   and  .../Non-Glaucoma/
           raw_dir/Test400/...
      B) Flat directory + CSV:
           raw_dir/images/
           raw_dir/labels.csv  (columns: ImgName, Label)
    """
    config = REFUGE_CONFIG
    print(f"\n[{config.name.upper()}] Preparing from: {raw_dir}")

    rows = []

    # Strategy A: directory-based labels (Glaucoma / Non-Glaucoma subdirs)
    dir_based_found = False
    for split_dir_name in [
        "Training400", "Validation400", "Test400",
        "training", "validation", "test",
        "train", "val", "test",
        "Training", "Validation", "Test",
        "TRAINING", "VALIDATION", "TEST",
    ]:
        split_dir = os.path.join(raw_dir, split_dir_name)
        if not os.path.isdir(split_dir):
            continue

        for label_dir_name, disease_label in [
            ("Glaucoma", 2),
            ("glaucoma", 2),
            ("GLAUCOMA", 2),
            ("Non-Glaucoma", 0),
            ("Non-glaucoma", 0),
            ("non-glaucoma", 0),
            ("NON-GLAUCOMA", 0),
            ("NonGlaucoma", 0),
            ("non_glaucoma", 0),
        ]:
            label_dir = os.path.join(split_dir, label_dir_name)
            if not os.path.isdir(label_dir):
                continue
            dir_based_found = True
            imgs = _find_images(label_dir)
            for img_path in imgs:
                rows.append({
                    "image_path": os.path.abspath(img_path),
                    "disease_label": disease_label,
                    "source": "REFUGE",
                    "severity_label": -1,
                    "cache_path": "",
                })

    if dir_based_found:
        print(f"  Found directory-based labels: {len(rows)} images")

    # Strategy B: CSV-based labels
    if not dir_based_found:
        labels_csv = _find_csv(raw_dir, [
            "labels.csv", "refuge_labels.csv", "glaucoma_labels.csv",
            "Fovea_location.xlsx",
        ])
        if labels_csv is not None:
            labels_df = pd.read_csv(labels_csv)
            print(f"  Labels CSV: {labels_csv} ({len(labels_df)} entries)")

            id_col = None
            for c in ["ImgName", "image", "Filename", "filename", "ID"]:
                if c in labels_df.columns:
                    id_col = c
                    break
            if id_col is None:
                id_col = labels_df.columns[0]

            label_col = None
            for c in ["Label", "Glaucoma", "label", "glaucoma", "diagnosis"]:
                if c in labels_df.columns:
                    label_col = c
                    break
            if label_col is None:
                label_col = labels_df.columns[1]

            # Find images
            img_dir = None
            for candidate in ["images", "Images", "fundus", "Fundus"]:
                d = os.path.join(raw_dir, candidate)
                if os.path.isdir(d):
                    img_dir = d
                    break
            if img_dir is None:
                img_dir = raw_dir

            for _, row in labels_df.iterrows():
                img_id = str(row[id_col]).strip()
                raw_label = int(row[label_col])
                disease_label = config.disease_label_map.get(raw_label, 0)

                img_path = None
                for ext in [".jpg", ".jpeg", ".png", ".tif", ".bmp", ""]:
                    candidate_path = os.path.join(img_dir, img_id + ext)
                    if os.path.exists(candidate_path):
                        img_path = os.path.abspath(candidate_path)
                        break

                if img_path is None:
                    continue

                rows.append({
                    "image_path": img_path,
                    "disease_label": disease_label,
                    "source": "REFUGE",
                    "severity_label": -1,
                    "cache_path": "",
                })

    # Strategy C: flat directory, no labels -- scan all images
    if not rows:
        all_imgs = _find_images(raw_dir)
        if all_imgs:
            print(f"  WARNING: No label structure found. Found {len(all_imgs)} images.")
            print("  Assigning all as Glaucoma (disease_label=2). Fix manually if incorrect.")
            for img_path in all_imgs:
                rows.append({
                    "image_path": os.path.abspath(img_path),
                    "disease_label": 2,
                    "source": "REFUGE",
                    "severity_label": -1,
                    "cache_path": "",
                })
        else:
            print(f"  ERROR: No images found in {raw_dir}")
            return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"  Saved: {output_csv}")
    _print_class_dist(df)
    return df


def prepare_adam(raw_dir: str, output_csv: str) -> pd.DataFrame:
    """
    Prepare ADAM / iChallenge-AMD.

    Handles two common structures:
      A) Directory-based labels:
           raw_dir/Training400/AMD/      and  .../Non-AMD/
      B) Flat directory + CSV:
           raw_dir/images/
           raw_dir/adam_labels.csv   (columns: imgName, label)
    """
    config = ADAM_CONFIG
    print(f"\n[{config.name.upper()}] Preparing from: {raw_dir}")

    rows = []

    # Strategy A: directory-based labels
    dir_based_found = False
    for split_dir_name in [
        "Training400", "Validation400", "Test400",
        "training", "validation", "test",
        "train", "val", "test",
        "Training", "Validation", "Test",
    ]:
        split_dir = os.path.join(raw_dir, split_dir_name)
        if not os.path.isdir(split_dir):
            continue

        for label_dir_name, disease_label in [
            ("AMD", 4),
            ("amd", 4),
            ("Non-AMD", 0),
            ("non-amd", 0),
            ("Non-amd", 0),
            ("NonAMD", 0),
            ("non_amd", 0),
        ]:
            label_dir = os.path.join(split_dir, label_dir_name)
            if not os.path.isdir(label_dir):
                continue
            dir_based_found = True
            imgs = _find_images(label_dir)
            for img_path in imgs:
                rows.append({
                    "image_path": os.path.abspath(img_path),
                    "disease_label": disease_label,
                    "source": "ADAM",
                    "severity_label": -1,
                    "cache_path": "",
                })

    if dir_based_found:
        print(f"  Found directory-based labels: {len(rows)} images")

    # Strategy B: CSV-based labels
    if not dir_based_found:
        labels_csv = _find_csv(raw_dir, [
            "adam_labels.csv", "labels.csv", "amd_labels.csv",
            "Classification_Labels.csv", "label.csv",
        ])
        if labels_csv is not None:
            labels_df = pd.read_csv(labels_csv)
            print(f"  Labels CSV: {labels_csv} ({len(labels_df)} entries)")

            id_col = None
            for c in ["imgName", "image", "Filename", "filename", "ID", "img"]:
                if c in labels_df.columns:
                    id_col = c
                    break
            if id_col is None:
                id_col = labels_df.columns[0]

            label_col = None
            for c in ["label", "Label", "AMD", "amd", "diagnosis"]:
                if c in labels_df.columns:
                    label_col = c
                    break
            if label_col is None:
                label_col = labels_df.columns[1]

            img_dir = None
            for candidate in ["images", "Images", "fundus", "Fundus", "imgs"]:
                d = os.path.join(raw_dir, candidate)
                if os.path.isdir(d):
                    img_dir = d
                    break
            if img_dir is None:
                img_dir = raw_dir

            for _, row in labels_df.iterrows():
                img_id = str(row[id_col]).strip()
                raw_label = int(row[label_col])
                disease_label = config.disease_label_map.get(raw_label, 0)

                img_path = None
                for ext in [".jpg", ".jpeg", ".png", ".tif", ".bmp", ""]:
                    candidate_path = os.path.join(img_dir, img_id + ext)
                    if os.path.exists(candidate_path):
                        img_path = os.path.abspath(candidate_path)
                        break

                if img_path is None:
                    continue

                rows.append({
                    "image_path": img_path,
                    "disease_label": disease_label,
                    "source": "ADAM",
                    "severity_label": -1,
                    "cache_path": "",
                })

    # Strategy C: flat scan
    if not rows:
        all_imgs = _find_images(raw_dir)
        if all_imgs:
            print(f"  WARNING: No label structure found. Found {len(all_imgs)} images.")
            print("  Assigning all as AMD (disease_label=4). Fix manually if incorrect.")
            for img_path in all_imgs:
                rows.append({
                    "image_path": os.path.abspath(img_path),
                    "disease_label": 4,
                    "source": "ADAM",
                    "severity_label": -1,
                    "cache_path": "",
                })
        else:
            print(f"  ERROR: No images found in {raw_dir}")
            return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"  Saved: {output_csv}")
    _print_class_dist(df)
    return df


def prepare_origa(raw_dir: str, output_csv: str) -> pd.DataFrame:
    """
    Prepare ORIGA.

    Expected structure:
        raw_dir/images/              (JPEG fundus images)
        raw_dir/origa_labels.csv     (columns: Filename, Glaucoma)
    """
    config = ORIGA_CONFIG
    print(f"\n[{config.name.upper()}] Preparing from: {raw_dir}")

    labels_csv = _find_csv(raw_dir, [
        "origa_labels.csv", "labels.csv", "ORIGA_labels.csv",
        "origa.csv", "glaucoma_labels.csv",
    ])

    # Also check for .xlsx
    if labels_csv is None:
        for name in ["origa_labels.xlsx", "ORIGA.xlsx", "labels.xlsx"]:
            path = os.path.join(raw_dir, name)
            if os.path.exists(path):
                try:
                    labels_df = pd.read_excel(path)
                    labels_csv = path
                    print(f"  Labels (Excel): {path}")
                    break
                except Exception:
                    continue

    img_dir = None
    for candidate in ["images", "Images", "fundus", "Fundus", "imgs"]:
        d = os.path.join(raw_dir, candidate)
        if os.path.isdir(d):
            img_dir = d
            break
    if img_dir is None:
        img_dir = raw_dir

    rows = []

    if labels_csv is not None:
        if labels_csv.endswith((".xlsx", ".xls")):
            labels_df = pd.read_excel(labels_csv)
        else:
            labels_df = pd.read_csv(labels_csv)
        print(f"  Labels: {labels_csv} ({len(labels_df)} entries)")

        id_col = None
        for c in ["Filename", "filename", "Image", "image", "ID", "File"]:
            if c in labels_df.columns:
                id_col = c
                break
        if id_col is None:
            id_col = labels_df.columns[0]

        label_col = None
        for c in ["Glaucoma", "glaucoma", "Label", "label", "diagnosis"]:
            if c in labels_df.columns:
                label_col = c
                break
        if label_col is None:
            label_col = labels_df.columns[1]

        found = 0
        missing = 0
        for _, row in labels_df.iterrows():
            img_id = str(row[id_col]).strip()
            raw_label = int(row[label_col])
            disease_label = config.disease_label_map.get(raw_label, 0)

            img_path = None
            for ext in [".jpg", ".jpeg", ".png", ".tif", ".bmp", ""]:
                candidate_path = os.path.join(img_dir, img_id + ext)
                if os.path.exists(candidate_path):
                    img_path = os.path.abspath(candidate_path)
                    break

            if img_path is None:
                missing += 1
                continue

            found += 1
            rows.append({
                "image_path": img_path,
                "disease_label": disease_label,
                "source": "ORIGA",
                "severity_label": -1,
                "cache_path": "",
            })
        print(f"  Found: {found} | Missing: {missing}")
    else:
        # No labels file -- scan images
        all_imgs = _find_images(img_dir)
        if all_imgs:
            print(f"  WARNING: No labels found. {len(all_imgs)} images discovered.")
            print("  Assigning all as Glaucoma (disease_label=2). Fix manually if incorrect.")
            for img_path in all_imgs:
                rows.append({
                    "image_path": os.path.abspath(img_path),
                    "disease_label": 2,
                    "source": "ORIGA",
                    "severity_label": -1,
                    "cache_path": "",
                })
        else:
            print(f"  ERROR: No images found in {raw_dir}")
            return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"  Saved: {output_csv}")
    _print_class_dist(df)
    return df


# Dispatcher map
PREPARE_FUNCTIONS = {
    "eyepacs": prepare_eyepacs,
    "messidor2": prepare_messidor2,
    "refuge": prepare_refuge,
    "adam": prepare_adam,
    "origa": prepare_origa,
}


# =====================================================================
# PREPROCESS AND CACHE
# =====================================================================


def preprocess_and_cache(
    csv_path: str,
    cache_dir: str,
    target_size: int = TARGET_SIZE,
    force: bool = False,
) -> Dict[str, int]:
    """
    Preprocess all images listed in a prepared CSV using the UNIFIED
    CLAHE pipeline and save as .npy arrays in cache_dir.

    Updates the 'cache_path' column in-place and re-saves the CSV.

    Parameters
    ----------
    csv_path : str
        Path to a prepared CSV (output of prepare_*).
    cache_dir : str
        Directory to store .npy cache files.
    target_size : int
        Output image dimension (square).
    force : bool
        If True, re-preprocess even if .npy already exists.

    Returns
    -------
    dict with keys: processed, skipped_cached, skipped_missing, errors, total.
    """
    os.makedirs(cache_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    stats = {
        "processed": 0,
        "skipped_cached": 0,
        "skipped_missing": 0,
        "errors": 0,
        "total": len(df),
    }

    cache_paths = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        img_path = row["image_path"]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        # Use a hash prefix to avoid filename collisions across datasets
        source_tag = str(row.get("source", "UNK")).lower()
        npy_name = f"{source_tag}_{stem}_v3.npy"
        npy_path = os.path.join(cache_dir, npy_name)
        cache_paths.append(npy_path)

        if os.path.exists(npy_path) and not force:
            stats["skipped_cached"] += 1
            continue

        if not os.path.exists(img_path):
            stats["skipped_missing"] += 1
            continue

        try:
            arr = unified_preprocess(img_path, target_size=target_size)
            if arr is None:
                stats["errors"] += 1
                continue
            np.save(npy_path, arr)
            stats["processed"] += 1
        except Exception as e:
            print(f"  ERROR processing {img_path}: {e}")
            stats["errors"] += 1

    df["cache_path"] = cache_paths
    df.to_csv(csv_path, index=False)

    print(f"\n  Preprocessing summary:")
    print(f"    Total:          {stats['total']}")
    print(f"    Processed:      {stats['processed']}")
    print(f"    Already cached: {stats['skipped_cached']}")
    print(f"    Missing files:  {stats['skipped_missing']}")
    print(f"    Errors:         {stats['errors']}")
    return stats


# =====================================================================
# MERGE ALL DATASETS
# =====================================================================


def merge_all_datasets(
    csv_paths: List[str],
    output_path: str,
    train_ratio: float = 0.70,
    calib_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Merge multiple prepared dataset CSVs into a unified corpus.

    Steps:
        1. Concatenate all CSVs
        2. Deduplicate by image_path
        3. Print class distribution
        4. Create stratified train/calib/test splits (70/15/15)
        5. Save merged CSV and split CSVs

    Parameters
    ----------
    csv_paths : list of str
        Paths to individual dataset CSVs.
    output_path : str
        Base directory for output files.
    train_ratio, calib_ratio, test_ratio : float
        Split proportions (must sum to 1.0).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    (merged_df, train_df, calib_df, test_df)
    """
    from sklearn.model_selection import train_test_split

    assert abs(train_ratio + calib_ratio + test_ratio - 1.0) < 1e-9, \
        f"Split ratios must sum to 1.0, got {train_ratio + calib_ratio + test_ratio}"

    os.makedirs(output_path, exist_ok=True)

    print("\n" + "=" * 70)
    print("  MERGING ALL DATASETS")
    print("=" * 70)

    # Load and concatenate
    dfs = []
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"  WARNING: {csv_path} not found, skipping.")
            continue
        df = pd.read_csv(csv_path)
        print(f"  Loaded: {csv_path} ({len(df)} rows)")
        dfs.append(df)

    if not dfs:
        raise ValueError("No valid CSV files found to merge.")

    merged = pd.concat(dfs, ignore_index=True)
    print(f"\n  Combined total: {len(merged)} rows")

    # Ensure consistent column set
    required_cols = ["image_path", "disease_label", "source", "severity_label", "cache_path"]
    for col in required_cols:
        if col not in merged.columns:
            if col == "severity_label":
                merged[col] = -1
            elif col == "cache_path":
                merged[col] = ""
            elif col == "source":
                merged[col] = "UNKNOWN"
            else:
                raise ValueError(f"Required column '{col}' missing from merged data.")

    # Deduplicate by image_path
    before_dedup = len(merged)
    merged = merged.drop_duplicates(subset=["image_path"], keep="first").reset_index(drop=True)
    n_dupes = before_dedup - len(merged)
    if n_dupes > 0:
        print(f"  Removed {n_dupes} duplicate image paths")
    print(f"  After deduplication: {len(merged)} rows")

    # Print source breakdown
    print(f"\n  --- Source Breakdown ---")
    for source, count in merged["source"].value_counts().sort_index().items():
        print(f"    {source:<12s}: {count:>6,}")

    # Print class distribution
    print(f"\n  --- Class Distribution ---")
    total = len(merged)
    max_count = 0
    min_count = float("inf")
    for label in sorted(merged["disease_label"].unique()):
        count = int((merged["disease_label"] == label).sum())
        pct = 100 * count / total
        name = CLASS_NAMES.get(label, f"Class_{label}")
        bar = "#" * int(pct / 2)
        print(f"    {label} {name:<14s}: {count:>6,}  ({pct:5.1f}%)  {bar}")
        max_count = max(max_count, count)
        min_count = min(min_count, count)
    print(f"    Imbalance ratio (max/min): {max_count / max(min_count, 1):.1f}:1")

    # Save merged CSV
    merged_csv = os.path.join(output_path, "merged_all_datasets.csv")
    merged.to_csv(merged_csv, index=False)
    print(f"\n  Merged CSV saved: {merged_csv}")

    # Stratified 3-way split
    print(f"\n  Creating stratified split ({train_ratio:.0%} / {calib_ratio:.0%} / {test_ratio:.0%})...")

    # Handle classes with very few samples (need at least 2 per split for stratification)
    label_counts = merged["disease_label"].value_counts()
    min_for_split = 3  # need at least 1 per split
    small_classes = label_counts[label_counts < min_for_split].index.tolist()
    if small_classes:
        print(f"  WARNING: Classes {small_classes} have fewer than {min_for_split} samples.")
        print(f"  These will be randomly distributed across splits.")

    train_df, temp_df = train_test_split(
        merged,
        test_size=(calib_ratio + test_ratio),
        stratify=merged["disease_label"],
        random_state=random_state,
    )

    calib_frac = calib_ratio / (calib_ratio + test_ratio)
    calib_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - calib_frac),
        stratify=temp_df["disease_label"],
        random_state=random_state,
    )

    train_df = train_df.reset_index(drop=True)
    calib_df = calib_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Save splits
    train_csv = os.path.join(output_path, "train_split.csv")
    calib_csv = os.path.join(output_path, "calib_split.csv")
    test_csv = os.path.join(output_path, "test_split.csv")
    train_df.to_csv(train_csv, index=False)
    calib_df.to_csv(calib_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    # Print split stats
    print()
    print("=" * 66)
    print("  STRATIFIED SPLIT -- CLASS DISTRIBUTION")
    print("=" * 66)
    print(f"  {'Class':<16s} {'Train':>8s} {'Calib':>8s} {'Test':>8s} {'Total':>8s}")
    print("  " + "-" * 54)
    tr_tot = ca_tot = te_tot = 0
    for lbl in sorted(CLASS_NAMES.keys()):
        tr = int((train_df["disease_label"] == lbl).sum())
        ca = int((calib_df["disease_label"] == lbl).sum())
        te = int((test_df["disease_label"] == lbl).sum())
        tot = tr + ca + te
        tr_tot += tr
        ca_tot += ca
        te_tot += te
        print(f"  {CLASS_NAMES[lbl]:<16s} {tr:>8,} {ca:>8,} {te:>8,} {tot:>8,}")
    print("  " + "-" * 54)
    total_n = len(train_df) + len(calib_df) + len(test_df)
    print(f"  {'TOTAL':<16s} {tr_tot:>8,} {ca_tot:>8,} {te_tot:>8,} {total_n:>8,}")
    print()
    print(f"  Split sizes : train={len(train_df):,}  calib={len(calib_df):,}  "
          f"test={len(test_df):,}")
    print(f"  Actual ratios: train={len(train_df)/total_n:.1%}  "
          f"calib={len(calib_df)/total_n:.1%}  "
          f"test={len(test_df)/total_n:.1%}")
    print()
    print(f"  Saved: {train_csv}")
    print(f"         {calib_csv}")
    print(f"         {test_csv}")

    return merged, train_df, calib_df, test_df


# =====================================================================
# INCLUDE EXISTING DATA
# =====================================================================


def include_existing_dataset(
    existing_csv: str,
    output_csv: str,
) -> pd.DataFrame:
    """
    Convert the existing final_unified_metadata.csv (APTOS + ODIR) into
    the standardised format used by this pipeline.

    The existing CSV uses 'dataset' as the source column name.
    This function renames it to 'source' for consistency and adds
    the cache_path column.
    """
    print(f"\n[EXISTING] Converting: {existing_csv}")
    if not os.path.exists(existing_csv):
        print(f"  ERROR: {existing_csv} not found")
        return pd.DataFrame()

    df = pd.read_csv(existing_csv)
    print(f"  Loaded {len(df)} rows")

    # Rename 'dataset' -> 'source' if needed
    if "dataset" in df.columns and "source" not in df.columns:
        df = df.rename(columns={"dataset": "source"})

    # Ensure all required columns exist
    if "cache_path" not in df.columns:
        df["cache_path"] = ""

    if "severity_label" not in df.columns:
        df["severity_label"] = -1

    # Ensure image_path column exists
    if "image_path" not in df.columns:
        raise ValueError("Existing CSV missing 'image_path' column")

    # Standardise column order
    df = df[["image_path", "disease_label", "source", "severity_label", "cache_path"]]

    df.to_csv(output_csv, index=False)
    print(f"  Saved: {output_csv}")
    _print_class_dist(df)
    return df


# =====================================================================
# HELPERS
# =====================================================================


def _print_class_dist(df: pd.DataFrame) -> None:
    """Print a compact class distribution summary."""
    if df.empty:
        print("  (empty dataframe)")
        return
    total = len(df)
    print(f"  Class distribution ({total} total):")
    for label in sorted(df["disease_label"].unique()):
        count = int((df["disease_label"] == label).sum())
        name = CLASS_NAMES.get(label, f"Class_{label}")
        pct = 100 * count / total
        print(f"    {label} {name:<14s}: {count:>6,}  ({pct:5.1f}%)")


def list_available_datasets() -> None:
    """Print all supported datasets with brief info."""
    print()
    print("=" * 70)
    print("  SUPPORTED DATASETS")
    print("=" * 70)
    for key, config in ALL_CONFIGS.items():
        print(f"\n  {key:<12s}  {config.description}")
        print(f"              Size: {config.approximate_size}")
        print(f"              URL:  {config.url}")
        print(f"              Labels: {config.label_type}")
    print()


# =====================================================================
# MAIN
# =====================================================================


def main():
    parser = argparse.ArgumentParser(
        description="RetinaSense Multi-Dataset Preparation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show all supported datasets
  python prepare_datasets.py --list

  # Print download instructions for a dataset
  python prepare_datasets.py --dataset refuge --instructions

  # Prepare EyePACS from downloaded raw data
  python prepare_datasets.py --dataset eyepacs --raw-dir ./data/eyepacs --output-dir ./data/

  # Prepare and preprocess (cache .npy files)
  python prepare_datasets.py --dataset refuge --raw-dir ./data/refuge --output-dir ./data/ --preprocess

  # Include existing APTOS+ODIR data
  python prepare_datasets.py --include-existing --output-dir ./data/

  # Merge all prepared CSVs into unified corpus with train/calib/test splits
  python prepare_datasets.py --merge --output-dir ./data/

  # Do everything: prepare all datasets, preprocess, merge
  python prepare_datasets.py --all --output-dir ./data/
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(ALL_CONFIGS.keys()),
        help="Dataset to prepare (eyepacs, messidor2, refuge, adam, origa)",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        help="Directory containing the raw downloaded dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/",
        help="Output directory for prepared CSVs and cache (default: ./data/)",
    )
    parser.add_argument(
        "--instructions",
        action="store_true",
        help="Print download instructions for the specified dataset",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all supported datasets",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Also preprocess images and save .npy cache after preparing CSV",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for .npy files (default: <output-dir>/preprocessed_cache_v3/)",
    )
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Include existing final_unified_metadata.csv (APTOS + ODIR)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge all prepared dataset CSVs in output-dir into unified corpus",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Prepare all datasets found in output-dir/raw/ subdirectories",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-preprocessing even if cache exists",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=TARGET_SIZE,
        help=f"Image resize target (default: {TARGET_SIZE})",
    )

    args = parser.parse_args()

    print()
    print("=" * 70)
    print("  RetinaSense -- Multi-Dataset Preparation Pipeline")
    print("=" * 70)

    cache_dir = args.cache_dir or os.path.join(args.output_dir, "preprocessed_cache_v3")

    # --list
    if args.list:
        list_available_datasets()
        return

    # --instructions
    if args.instructions:
        if args.dataset:
            download_instructions(ALL_CONFIGS[args.dataset])
        else:
            for config in ALL_CONFIGS.values():
                download_instructions(config)
        return

    # --include-existing (convert current APTOS+ODIR CSV)
    if args.include_existing:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        existing_csv = os.path.join(script_dir, "final_unified_metadata.csv")
        output_csv = os.path.join(args.output_dir, "prepared_existing.csv")
        os.makedirs(args.output_dir, exist_ok=True)
        df = include_existing_dataset(existing_csv, output_csv)
        if args.preprocess and not df.empty:
            preprocess_and_cache(output_csv, cache_dir, target_size=args.target_size, force=args.force)
        if not args.merge and not args.all:
            return

    # --dataset (prepare a single dataset)
    if args.dataset and not args.all:
        if args.raw_dir is None:
            print(f"\n  ERROR: --raw-dir is required when preparing a dataset.")
            print(f"  Usage: python prepare_datasets.py --dataset {args.dataset} --raw-dir <path>")
            sys.exit(1)

        os.makedirs(args.output_dir, exist_ok=True)
        output_csv = os.path.join(args.output_dir, f"prepared_{args.dataset}.csv")
        prepare_fn = PREPARE_FUNCTIONS[args.dataset]
        df = prepare_fn(args.raw_dir, output_csv)

        if args.preprocess and not df.empty:
            preprocess_and_cache(output_csv, cache_dir, target_size=args.target_size, force=args.force)
        return

    # --all (prepare all datasets found in standard locations)
    if args.all:
        os.makedirs(args.output_dir, exist_ok=True)
        prepared_csvs = []

        # Include existing data first
        script_dir = os.path.dirname(os.path.abspath(__file__))
        existing_csv = os.path.join(script_dir, "final_unified_metadata.csv")
        if os.path.exists(existing_csv):
            output_csv = os.path.join(args.output_dir, "prepared_existing.csv")
            df = include_existing_dataset(existing_csv, output_csv)
            if not df.empty:
                prepared_csvs.append(output_csv)
                if args.preprocess:
                    preprocess_and_cache(output_csv, cache_dir, target_size=args.target_size, force=args.force)

        # Try to prepare each dataset from its expected location
        for ds_name, prepare_fn in PREPARE_FUNCTIONS.items():
            raw_dir = os.path.join(args.output_dir, ds_name)
            if not os.path.isdir(raw_dir):
                # Also check parent directory
                parent_raw = os.path.join(os.path.dirname(args.output_dir), ds_name)
                if os.path.isdir(parent_raw):
                    raw_dir = parent_raw
                else:
                    print(f"\n  [{ds_name.upper()}] Skipping -- raw directory not found: {raw_dir}")
                    continue

            output_csv = os.path.join(args.output_dir, f"prepared_{ds_name}.csv")
            try:
                df = prepare_fn(raw_dir, output_csv)
                if not df.empty:
                    prepared_csvs.append(output_csv)
                    if args.preprocess:
                        preprocess_and_cache(output_csv, cache_dir, target_size=args.target_size, force=args.force)
            except Exception as e:
                print(f"\n  [{ds_name.upper()}] ERROR: {e}")
                continue

        # Auto-merge if we prepared anything
        if prepared_csvs:
            merge_all_datasets(prepared_csvs, args.output_dir)
        else:
            print("\n  No datasets were prepared. Download datasets first:")
            for config in ALL_CONFIGS.values():
                print(f"    {config.name:<12s}: {config.url}")
        return

    # --merge
    if args.merge:
        # Find all prepared_*.csv files in output_dir
        prepared_csvs = sorted(
            str(p) for p in Path(args.output_dir).glob("prepared_*.csv")
        )
        if not prepared_csvs:
            print(f"\n  No prepared_*.csv files found in {args.output_dir}")
            print("  Prepare datasets first with --dataset or --all")
            sys.exit(1)

        print(f"\n  Found {len(prepared_csvs)} prepared CSVs:")
        for p in prepared_csvs:
            print(f"    {p}")

        merge_all_datasets(prepared_csvs, args.output_dir)
        return

    # No action specified
    parser.print_help()


if __name__ == "__main__":
    main()
