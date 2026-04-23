#!/usr/bin/env python3
"""
rebuild_faiss_full.py -- Rebuild FAISS Index with ALL 5 Classes (DANN-v3)
=========================================================================
The existing FAISS index only has Normal + DR (7,429 vectors).
This script rebuilds from scratch using the DANN-v3 model to embed
ALL 8,241 training samples across all 5 classes.

Uses IndexFlatIP (inner product = cosine similarity after L2 normalization).

Usage:
    python rebuild_faiss_full.py
    python rebuild_faiss_full.py --batch-size 128 --device cuda
    python rebuild_faiss_full.py --device cpu --batch-size 16
"""

import os
import sys
import json
import csv
import time
import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import faiss
import timm
from torchvision import transforms
from tqdm import tqdm

# ================================================================
# PATHS (all relative to script directory)
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(BASE_DIR, "data", "train_split_expanded.csv")
NORM_PATH = os.path.join(BASE_DIR, "configs", "fundus_norm_stats_unified.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_v3", "retrieval")

# Model candidates (prefer DANN-v3 -> v2 -> v1 -> original)
_MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "outputs_v3", "dann_v3", "best_model.pth"),
    os.path.join(BASE_DIR, "outputs_v3", "dann_v2", "best_model.pth"),
    os.path.join(BASE_DIR, "outputs_v3", "dann", "best_model.pth"),
    os.path.join(BASE_DIR, "outputs_v3", "best_model.pth"),
]

# Cache directories (prefer unified -> v3)
CACHE_DIRS = [
    os.path.join(BASE_DIR, "preprocessed_cache_unified"),
    os.path.join(BASE_DIR, "preprocessed_cache_v3"),
]

CLASS_NAMES = ["Normal", "Diabetes/DR", "Glaucoma", "Cataract", "AMD"]
FEAT_DIM = 768


# ================================================================
# MODEL
# ================================================================
class MultiTaskViT(nn.Module):
    """MultiTaskViT architecture matching app.py (no domain head)."""

    def __init__(self, n_disease=5, n_severity=5, drop=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch16_224", pretrained=False, num_classes=0
        )
        self.drop = nn.Dropout(drop)
        self.disease_head = nn.Sequential(
            nn.Linear(768, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_disease),
        )
        self.severity_head = nn.Sequential(
            nn.Linear(768, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_severity),
        )

    def forward(self, x):
        f = self.backbone(x)
        f = self.drop(f)
        return self.disease_head(f), self.severity_head(f)


def load_model(model_path, device):
    """Load MultiTaskViT from DANN checkpoint, filtering domain_head/grl keys."""
    print(f"Loading model from {model_path}")
    model = MultiTaskViT().to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"]
    filtered = {
        k: v for k, v in state_dict.items()
        if not k.startswith("domain_head") and not k.startswith("grl")
    }
    removed = len(state_dict) - len(filtered)
    if removed:
        print(f"  Filtered out {removed} DANN keys (domain_head/grl)")
    result = model.load_state_dict(filtered, strict=False)
    if result.missing_keys:
        print(f"  WARNING: {len(result.missing_keys)} missing keys: {result.missing_keys}")
    model.eval()
    epoch = ckpt.get("epoch", "?")
    val_acc = ckpt.get("val_acc", 0.0)
    print(f"  Epoch {epoch}, val_acc={val_acc:.2f}%")
    return model


# ================================================================
# DATA LOADING
# ================================================================
def load_norm_stats():
    """Load fundus normalization stats from config."""
    with open(NORM_PATH) as f:
        stats = json.load(f)
    return stats["mean_rgb"], stats["std_rgb"]


def resolve_cache_path(csv_cache_path):
    """Resolve cache path: try unified cache first, then v3 cache, then CSV path."""
    stem = os.path.basename(csv_cache_path)
    for cache_dir in CACHE_DIRS:
        candidate = os.path.join(cache_dir, stem)
        if os.path.exists(candidate):
            return candidate
    # Fallback: try the CSV path as-is (relative to BASE_DIR)
    abs_path = os.path.join(BASE_DIR, csv_cache_path.lstrip("./"))
    if os.path.exists(abs_path):
        return abs_path
    return None


def load_training_samples():
    """Load all training samples from the CSV, resolving cache paths."""
    print(f"Loading training data from {TRAIN_CSV}")
    samples = []
    missing = 0
    with open(TRAIN_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cache_path = resolve_cache_path(row["cache_path"])
            if cache_path is None:
                missing += 1
                continue
            samples.append({
                "image_path": row["image_path"],
                "cache_path": cache_path,
                "label": int(row["disease_label"]),
                "class_name": CLASS_NAMES[int(row["disease_label"])],
                "source": row["source"],
            })
    print(f"  Loaded {len(samples)} samples ({missing} missing cache files)")
    return samples


def load_cached_image(cache_path, transform):
    """Load a preprocessed .npy image and apply normalization transform."""
    img = np.load(cache_path)  # (224, 224, 3), uint8
    # Transform expects PIL or uint8 numpy -> ToTensor -> Normalize
    tensor = transform(img)  # (3, 224, 224)
    return tensor


# ================================================================
# EMBEDDING EXTRACTION
# ================================================================
@torch.no_grad()
def extract_all_embeddings(model, samples, transform, device, batch_size=64,
                           use_amp=True):
    """Extract L2-normalized backbone embeddings for all samples.

    Args:
        model: MultiTaskViT model (eval mode)
        samples: list of dicts with 'cache_path' key
        transform: torchvision transform for normalization
        device: torch device
        batch_size: batch size for inference
        use_amp: whether to use automatic mixed precision

    Returns:
        embeddings: numpy array of shape (N, 768), L2-normalized
    """
    model.eval()
    all_embeddings = []
    n = len(samples)
    num_batches = (n + batch_size - 1) // batch_size

    amp_enabled = use_amp and device.type == "cuda"
    amp_dtype = torch.float16 if amp_enabled else torch.float32

    print(f"Extracting embeddings: {n} samples, batch_size={batch_size}, "
          f"AMP={'ON' if amp_enabled else 'OFF'}")

    for batch_idx in tqdm(range(num_batches), desc="Extracting embeddings"):
        start = batch_idx * batch_size
        end = min(start + batch_size, n)
        batch_samples = samples[start:end]

        # Load batch
        tensors = []
        for s in batch_samples:
            t = load_cached_image(s["cache_path"], transform)
            tensors.append(t)
        batch_tensor = torch.stack(tensors).to(device)  # (B, 3, 224, 224)

        # Extract backbone features
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_enabled):
            features = model.backbone(batch_tensor)  # (B, 768)

        # Move to CPU and convert to float32
        features = features.float().cpu().numpy()
        all_embeddings.append(features)

    # Concatenate all
    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    assert embeddings.shape == (n, FEAT_DIM), \
        f"Expected ({n}, {FEAT_DIM}), got {embeddings.shape}"

    # L2 normalize
    faiss.normalize_L2(embeddings)
    print(f"  Embeddings shape: {embeddings.shape}, L2-normalized")

    return embeddings


# ================================================================
# FAISS INDEX BUILDING
# ================================================================
def build_faiss_index(embeddings):
    """Build a FAISS IndexFlatIP (cosine similarity after L2 normalization).

    Inner product on L2-normalized vectors = cosine similarity.
    """
    print(f"Building FAISS IndexFlatIP with {embeddings.shape[0]} vectors, "
          f"dim={embeddings.shape[1]}")
    index = faiss.IndexFlatIP(FEAT_DIM)
    index.add(embeddings)
    print(f"  Index built: {index.ntotal} vectors")
    return index


def save_index_and_metadata(index, samples, output_dir):
    """Save FAISS index and metadata JSON to disk."""
    os.makedirs(output_dir, exist_ok=True)

    # Save index
    index_path = os.path.join(output_dir, "index_flat_ip.faiss")
    faiss.write_index(index, index_path)
    size_mb = os.path.getsize(index_path) / (1024 * 1024)
    print(f"  Saved index: {index_path} ({size_mb:.1f} MB)")

    # Build metadata with relative paths for portability
    metadata = []
    for i, s in enumerate(samples):
        # Store cache_path relative to BASE_DIR
        rel_cache = os.path.relpath(s["cache_path"], BASE_DIR)
        metadata.append({
            "index": i,
            "image_path": s["image_path"],
            "cache_path": rel_cache,
            "label": s["label"],
            "class_name": s["class_name"],
            "source": s["source"],
        })

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=None)  # compact for speed
    size_mb = os.path.getsize(meta_path) / (1024 * 1024)
    print(f"  Saved metadata: {meta_path} ({size_mb:.1f} MB, {len(metadata)} entries)")

    return index_path, meta_path


# ================================================================
# MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Rebuild FAISS index with ALL 5 classes using DANN-v3 embeddings"
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Override model checkpoint path (default: auto-detect DANN-v3/v2/v1)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for embedding extraction (default: 64)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: cuda or cpu (default: auto-detect)"
    )
    parser.add_argument(
        "--no-amp", action="store_true",
        help="Disable automatic mixed precision"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory (default: outputs_v3/retrieval)"
    )
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model path
    if args.model_path:
        model_path = args.model_path
        if not os.path.isabs(model_path):
            model_path = os.path.join(BASE_DIR, model_path)
    else:
        model_path = next(
            (p for p in _MODEL_CANDIDATES if os.path.exists(p)),
            _MODEL_CANDIDATES[0]
        )
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)

    # Output dir
    output_dir = args.output_dir or OUTPUT_DIR
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(BASE_DIR, output_dir)

    # ---- Step 1: Load model ----
    t0 = time.time()
    model = load_model(model_path, device)

    # ---- Step 2: Load training samples ----
    samples = load_training_samples()
    if not samples:
        print("ERROR: No training samples found")
        sys.exit(1)

    # ---- Step 3: Prepare transform ----
    mean, std = load_norm_stats()
    print(f"Normalization: mean={mean}, std={std}")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # ---- Step 4: Extract embeddings ----
    embeddings = extract_all_embeddings(
        model, samples, transform, device,
        batch_size=args.batch_size,
        use_amp=not args.no_amp,
    )

    # ---- Step 5: Build FAISS index ----
    index = build_faiss_index(embeddings)

    # ---- Step 6: Save ----
    index_path, meta_path = save_index_and_metadata(index, samples, output_dir)

    # ---- Step 7: Print distributions ----
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"FAISS INDEX REBUILT SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"  Total vectors: {index.ntotal}")
    print(f"  Dimension: {FEAT_DIM}")
    print(f"  Index type: IndexFlatIP (cosine similarity)")
    print(f"  Time: {elapsed:.1f}s")
    print(f"\n  Class distribution:")
    class_counts = Counter(s["label"] for s in samples)
    for label in sorted(class_counts.keys()):
        name = CLASS_NAMES[label]
        count = class_counts[label]
        pct = 100.0 * count / len(samples)
        print(f"    {name:15s}: {count:5d} ({pct:5.1f}%)")

    print(f"\n  Source distribution:")
    source_counts = Counter(s["source"] for s in samples)
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / len(samples)
        print(f"    {source:15s}: {count:5d} ({pct:5.1f}%)")

    print(f"\n  Output files:")
    print(f"    Index: {index_path}")
    print(f"    Metadata: {meta_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
