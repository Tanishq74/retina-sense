#!/usr/bin/env python3
"""
Update FAISS retrieval index with MESSIDOR-2 images.

Loads the existing FAISS index and metadata, extracts ViT backbone embeddings
for all MESSIDOR-2 entries from the expanded training CSV, and appends them
to the index and metadata.

Designed to run on CPU with batch_size=8 to keep memory low.
"""

import os, sys, json, csv, time
import numpy as np
import torch
import torch.nn as nn
import faiss
import timm

# ================================================================
# PATHS
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "outputs_v3/retrieval/index_flat_l2.faiss")
META_PATH = os.path.join(BASE_DIR, "outputs_v3/retrieval/metadata.json")
CSV_PATH = os.path.join(BASE_DIR, "data/train_split_expanded.csv")
_model_candidates = [
    os.path.join(BASE_DIR, "outputs_v3/dann_v3/best_model.pth"),
    os.path.join(BASE_DIR, "outputs_v3/dann_v2/best_model.pth"),
    os.path.join(BASE_DIR, "outputs_v3/dann/best_model.pth"),
]
MODEL_PATH = next((p for p in _model_candidates if os.path.exists(p)), _model_candidates[-1])
CACHE_DIR = os.path.join(BASE_DIR, "preprocessed_cache_unified")

DEVICE = torch.device("cpu")
BATCH_SIZE = 8
CLASS_NAMES = ["Normal", "Diabetes/DR", "Glaucoma", "Cataract", "AMD"]

# ================================================================
# MODEL (same as app.py -- MultiTaskViT, filter DANN keys)
# ================================================================
class MultiTaskViT(nn.Module):
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


def load_model():
    """Load MultiTaskViT from DANN checkpoint, filtering domain_head/grl keys."""
    print(f"Loading model from {MODEL_PATH} ...")
    model = MultiTaskViT().to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
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
        print(f"  WARNING: {len(result.missing_keys)} missing keys")
    model.eval()
    print("  Model loaded and set to eval mode.")
    return model


def extract_backbone_embedding(model, batch_tensor):
    """Extract 768-dim backbone embeddings (before disease/severity heads)."""
    with torch.no_grad():
        features = model.backbone(batch_tensor)  # (B, 768)
    return features.numpy()


# ================================================================
# MAIN
# ================================================================
def main():
    t0 = time.time()

    # --- Load existing index and metadata ---
    print(f"\n[1/5] Loading existing FAISS index from {INDEX_PATH}")
    index = faiss.read_index(INDEX_PATH)
    print(f"  Existing vectors: {index.ntotal}, dimension: {index.d}")
    assert index.d == 768, f"Expected dim 768, got {index.d}"

    print(f"\n[2/5] Loading existing metadata from {META_PATH}")
    with open(META_PATH) as f:
        metadata = json.load(f)
    print(f"  Existing metadata entries: {len(metadata)}")
    assert len(metadata) == index.ntotal, (
        f"Metadata ({len(metadata)}) != index ({index.ntotal})"
    )

    # Check if MESSIDOR2 entries already exist
    existing_sources = set(e.get("dataset_source", "") for e in metadata)
    if "MESSIDOR2" in existing_sources:
        n_existing = sum(1 for e in metadata if e.get("dataset_source") == "MESSIDOR2")
        print(f"  WARNING: {n_existing} MESSIDOR2 entries already in metadata!")
        print("  Aborting to avoid duplicates. Remove them first if re-indexing.")
        sys.exit(1)

    # --- Read MESSIDOR2 entries from CSV ---
    print(f"\n[3/5] Reading MESSIDOR2 entries from {CSV_PATH}")
    messidor_entries = []
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["source"] == "MESSIDOR2":
                # Build absolute cache path
                cache_rel = row["cache_path"]
                if not os.path.isabs(cache_rel):
                    cache_abs = os.path.join(BASE_DIR, cache_rel)
                else:
                    cache_abs = cache_rel
                messidor_entries.append({
                    "image_path": row["image_path"],
                    "cache_path": cache_abs,
                    "disease_label": int(row["disease_label"]),
                    "source": row["source"],
                })

    print(f"  Found {len(messidor_entries)} MESSIDOR2 entries in CSV")

    # Verify cache files exist
    missing = []
    for entry in messidor_entries:
        if not os.path.exists(entry["cache_path"]):
            missing.append(entry["cache_path"])
    if missing:
        print(f"  WARNING: {len(missing)} cache files missing! First 5:")
        for p in missing[:5]:
            print(f"    {p}")
        # Filter to only existing
        messidor_entries = [e for e in messidor_entries if os.path.exists(e["cache_path"])]
        print(f"  Proceeding with {len(messidor_entries)} entries that have cache files")

    if not messidor_entries:
        print("  No valid MESSIDOR2 entries found. Aborting.")
        sys.exit(1)

    # --- Load model and extract embeddings ---
    print(f"\n[4/5] Extracting embeddings (batch_size={BATCH_SIZE}, device={DEVICE})")
    model = load_model()

    # Load norm stats for proper normalization
    norm_path = os.path.join(BASE_DIR, "configs/fundus_norm_stats_unified.json")
    with open(norm_path) as f:
        ns = json.load(f)
    norm_mean = np.array(ns["mean_rgb"], dtype=np.float32)
    norm_std = np.array(ns["std_rgb"], dtype=np.float32)
    print(f"  Norm stats: mean={norm_mean.tolist()}, std={norm_std.tolist()}")

    all_embeddings = []
    n_total = len(messidor_entries)
    n_batches = (n_total + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(n_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, n_total)
        batch_entries = messidor_entries[start:end]

        # Load and normalize cached images
        tensors = []
        for entry in batch_entries:
            arr = np.load(entry["cache_path"])  # (224, 224, 3) uint8 or float
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0
            # Normalize with fundus stats
            arr = (arr - norm_mean) / norm_std
            # HWC -> CHW
            tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float()
            tensors.append(tensor)

        batch_tensor = torch.stack(tensors).to(DEVICE)  # (B, 3, 224, 224)
        embeddings = extract_backbone_embedding(model, batch_tensor)  # (B, 768)
        all_embeddings.append(embeddings)

        if (batch_idx + 1) % 25 == 0 or batch_idx == n_batches - 1:
            elapsed = time.time() - t0
            print(f"  Batch {batch_idx+1}/{n_batches} "
                  f"({end}/{n_total} images, {elapsed:.1f}s elapsed)")

    all_embeddings = np.vstack(all_embeddings).astype(np.float32)  # (N, 768)
    print(f"  Extracted {all_embeddings.shape[0]} embeddings, dim={all_embeddings.shape[1]}")

    # --- Add to FAISS index and update metadata ---
    print(f"\n[5/5] Adding to FAISS index and updating metadata")
    next_index = len(metadata)  # Continue from last index

    # Add vectors to FAISS index
    index.add(all_embeddings)
    print(f"  FAISS index: {index.ntotal} vectors (was {next_index})")

    # Build new metadata entries
    new_metadata = []
    for i, entry in enumerate(messidor_entries):
        label = entry["disease_label"]
        class_name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else f"Unknown({label})"
        new_metadata.append({
            "index": next_index + i,
            "image_path": entry["image_path"],
            "cache_path": entry["cache_path"],
            "label": label,
            "class_name": class_name,
            "dataset_source": entry["source"],
        })

    metadata.extend(new_metadata)
    assert len(metadata) == index.ntotal, (
        f"Metadata ({len(metadata)}) != index ({index.ntotal}) after update"
    )

    # Save updated index
    faiss.write_index(index, INDEX_PATH)
    print(f"  Saved FAISS index to {INDEX_PATH}")

    # Save updated metadata
    with open(META_PATH, "w") as f:
        json.dump(metadata, f)
    print(f"  Saved metadata to {META_PATH} ({len(metadata)} entries)")

    # --- Summary ---
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed:.1f}s")
    print(f"  Vectors added:  {all_embeddings.shape[0]}")
    print(f"  Total vectors:  {index.ntotal}")
    print(f"  Total metadata: {len(metadata)}")

    # Label distribution of new entries
    from collections import Counter
    label_dist = Counter(e["label"] for e in new_metadata)
    print(f"  New entries by class:")
    for label in sorted(label_dist):
        print(f"    {CLASS_NAMES[label]}: {label_dist[label]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
