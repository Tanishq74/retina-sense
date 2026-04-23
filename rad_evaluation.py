#!/usr/bin/env python3
"""
rad_evaluation.py -- Retrieval-Augmented Diagnosis (RAD) Evaluation Framework
==============================================================================
Evaluates the FAISS-based retrieval system on the test set:

Metrics computed:
  - Recall@K (K=1,3,5,10): fraction of queries with >= 1 same-class retrieval
  - Precision@K: fraction of retrieved cases matching true class
  - Mean Average Precision (MAP)
  - Class-Match Rate: per-class heatmap of retrieval patterns
  - Agreement Score: does kNN majority agree with model prediction?
  - Retrieval-Augmented Accuracy: kNN vote combined with model logits

Outputs:
  - outputs_v3/retrieval/rad_evaluation_results.json
  - outputs_v3/retrieval/retrieval_recall_at_k.png
  - outputs_v3/retrieval/class_match_heatmap.png
  - outputs_v3/retrieval/agreement_analysis.png

Usage:
    python rad_evaluation.py
    python rad_evaluation.py --k-values 1 3 5 10 20 --alpha 0.6
    python rad_evaluation.py --device cpu --batch-size 16
"""

import os
import sys
import json
import csv
import time
import argparse
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
import timm
from torchvision import transforms
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ================================================================
# PATHS
# ================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_CSV = os.path.join(BASE_DIR, "data", "test_split.csv")
NORM_PATH = os.path.join(BASE_DIR, "configs", "fundus_norm_stats_unified.json")
TEMP_PATH = os.path.join(BASE_DIR, "configs", "temperature.json")
RETRIEVAL_DIR = os.path.join(BASE_DIR, "outputs_v3", "retrieval")

_MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "outputs_v3", "dann_v3", "best_model.pth"),
    os.path.join(BASE_DIR, "outputs_v3", "dann_v2", "best_model.pth"),
    os.path.join(BASE_DIR, "outputs_v3", "dann", "best_model.pth"),
    os.path.join(BASE_DIR, "outputs_v3", "best_model.pth"),
]

CACHE_DIRS = [
    os.path.join(BASE_DIR, "preprocessed_cache_unified"),
    os.path.join(BASE_DIR, "preprocessed_cache_v3"),
]

CLASS_NAMES = ["Normal", "Diabetes/DR", "Glaucoma", "Cataract", "AMD"]
NUM_CLASSES = 5
FEAT_DIM = 768


# ================================================================
# MODEL
# ================================================================
class MultiTaskViT(nn.Module):
    """MultiTaskViT architecture matching app.py."""

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
        print(f"  Filtered out {removed} DANN keys")
    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model


# ================================================================
# DATA LOADING
# ================================================================
def resolve_cache_path(csv_cache_path):
    """Resolve cache path: try unified cache first, then v3 cache."""
    stem = os.path.basename(csv_cache_path)
    for cache_dir in CACHE_DIRS:
        candidate = os.path.join(cache_dir, stem)
        if os.path.exists(candidate):
            return candidate
    abs_path = os.path.join(BASE_DIR, csv_cache_path.lstrip("./"))
    if os.path.exists(abs_path):
        return abs_path
    return None


def load_test_samples():
    """Load test samples from CSV."""
    print(f"Loading test data from {TEST_CSV}")
    samples = []
    missing = 0
    with open(TEST_CSV) as f:
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
    print(f"  Loaded {len(samples)} test samples ({missing} missing)")
    return samples


def load_faiss_index():
    """Load FAISS index and metadata. Prefers IndexFlatIP, falls back to L2."""
    # Try IndexFlatIP first (rebuilt), then IndexFlatL2 (legacy)
    ip_path = os.path.join(RETRIEVAL_DIR, "index_flat_ip.faiss")
    l2_path = os.path.join(RETRIEVAL_DIR, "index_flat_l2.faiss")
    meta_path = os.path.join(RETRIEVAL_DIR, "metadata.json")

    if os.path.exists(ip_path):
        index_path = ip_path
        index_type = "IP"
    elif os.path.exists(l2_path):
        index_path = l2_path
        index_type = "L2"
    else:
        print("ERROR: No FAISS index found. Run rebuild_faiss_full.py first.")
        sys.exit(1)

    print(f"Loading FAISS index from {index_path} (type: {index_type})")
    index = faiss.read_index(index_path)
    print(f"  Index loaded: {index.ntotal} vectors")

    if not os.path.exists(meta_path):
        print(f"ERROR: Metadata not found at {meta_path}")
        sys.exit(1)
    with open(meta_path) as f:
        metadata = json.load(f)
    print(f"  Metadata loaded: {len(metadata)} entries")

    # Verify class distribution
    class_dist = Counter(m["class_name"] for m in metadata)
    print(f"  Index class distribution: {dict(class_dist)}")

    return index, metadata, index_type


# ================================================================
# EMBEDDING EXTRACTION
# ================================================================
@torch.no_grad()
def extract_test_embeddings_and_logits(model, samples, transform, device,
                                       batch_size=64, use_amp=True):
    """Extract backbone embeddings and disease logits for test samples.

    Returns:
        embeddings: (N, 768) L2-normalized
        logits: (N, 5) raw logits
        labels: (N,) integer labels
    """
    model.eval()
    all_embeddings = []
    all_logits = []
    all_labels = []
    n = len(samples)
    num_batches = (n + batch_size - 1) // batch_size

    amp_enabled = use_amp and device.type == "cuda"
    amp_dtype = torch.float16 if amp_enabled else torch.float32

    for batch_idx in tqdm(range(num_batches), desc="Test embeddings"):
        start = batch_idx * batch_size
        end = min(start + batch_size, n)
        batch_samples = samples[start:end]

        tensors = []
        for s in batch_samples:
            img = np.load(s["cache_path"])
            t = transform(img)
            tensors.append(t)
        batch_tensor = torch.stack(tensors).to(device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_enabled):
            features = model.backbone(batch_tensor)
            f_dropped = model.drop(features)
            disease_logits = model.disease_head(f_dropped)

        all_embeddings.append(features.float().cpu().numpy())
        all_logits.append(disease_logits.float().cpu().numpy())
        all_labels.extend([s["label"] for s in batch_samples])

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    logits = np.concatenate(all_logits, axis=0).astype(np.float32)
    labels = np.array(all_labels, dtype=np.int64)

    # L2 normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    return embeddings, logits, labels


# ================================================================
# RETRIEVAL METRICS
# ================================================================
def compute_retrieval_metrics(index, metadata, test_embeddings, test_labels,
                              test_sources, k_values, index_type="IP"):
    """Compute all retrieval metrics at various K values.

    Returns dict with all computed metrics.
    """
    max_k = max(k_values)
    n_test = len(test_labels)

    print(f"\nSearching index for top-{max_k} neighbors ({n_test} queries)...")
    distances, indices = index.search(test_embeddings, max_k)

    # Get retrieved labels
    meta_labels = np.array([m["label"] for m in metadata])
    meta_sources = [m.get("source", "unknown") for m in metadata]

    # ---- Per-query results ----
    retrieved_labels = np.zeros((n_test, max_k), dtype=np.int64)
    retrieved_sources = []
    for i in range(n_test):
        row_sources = []
        for j in range(max_k):
            idx = indices[i, j]
            if 0 <= idx < len(metadata):
                retrieved_labels[i, j] = meta_labels[idx]
                row_sources.append(meta_sources[idx])
            else:
                retrieved_labels[i, j] = -1
                row_sources.append("unknown")
        retrieved_sources.append(row_sources)

    # Convert distances to similarity scores
    if index_type == "IP":
        # Inner product: higher is better, already cosine similarity
        similarities = distances
    else:
        # L2: lower is better, convert to similarity
        similarities = 1.0 / (1.0 + distances)

    results = {
        "k_values": k_values,
        "n_test": n_test,
        "n_index": index.ntotal,
    }

    # ---- Recall@K and Precision@K ----
    recall_at_k = {}
    precision_at_k = {}
    for k in k_values:
        top_k_labels = retrieved_labels[:, :k]
        matches = (top_k_labels == test_labels[:, None])
        recall = float(np.mean(matches.any(axis=1)))
        precision = float(np.mean(matches.sum(axis=1) / k))
        recall_at_k[k] = recall
        precision_at_k[k] = precision

    results["recall_at_k"] = {str(k): v for k, v in recall_at_k.items()}
    results["precision_at_k"] = {str(k): v for k, v in precision_at_k.items()}

    # ---- Mean Average Precision (MAP) ----
    aps = []
    for i in range(n_test):
        true_label = test_labels[i]
        relevant = (retrieved_labels[i] == true_label)
        if not relevant.any():
            aps.append(0.0)
            continue
        cum_relevant = np.cumsum(relevant).astype(np.float64)
        precisions = cum_relevant / np.arange(1, max_k + 1)
        ap = float(np.sum(precisions * relevant) / relevant.sum())
        aps.append(ap)
    results["map"] = float(np.mean(aps))

    # ---- Per-class metrics ----
    per_class = {}
    for c in range(NUM_CLASSES):
        mask = (test_labels == c)
        if mask.sum() == 0:
            continue
        class_name = CLASS_NAMES[c]
        class_results = {"n_samples": int(mask.sum())}
        for k in k_values:
            top_k = retrieved_labels[mask, :k]
            matches_c = (top_k == c)
            class_results[f"recall@{k}"] = float(np.mean(matches_c.any(axis=1)))
            class_results[f"precision@{k}"] = float(
                np.mean(matches_c.sum(axis=1) / k)
            )
        # Class AP
        class_aps = [aps[i] for i in range(n_test) if test_labels[i] == c]
        class_results["ap"] = float(np.mean(class_aps)) if class_aps else 0.0
        per_class[class_name] = class_results
    results["per_class"] = per_class

    # ---- Per-source metrics ----
    per_source = {}
    unique_sources = sorted(set(test_sources))
    for src in unique_sources:
        mask = np.array([s == src for s in test_sources])
        if mask.sum() == 0:
            continue
        src_results = {"n_samples": int(mask.sum())}
        for k in k_values:
            top_k = retrieved_labels[mask, :k]
            matches_s = (top_k == test_labels[mask][:, None])
            src_results[f"recall@{k}"] = float(np.mean(matches_s.any(axis=1)))
            src_results[f"precision@{k}"] = float(
                np.mean(matches_s.sum(axis=1) / k)
            )
        per_source[src] = src_results
    results["per_source"] = per_source

    # ---- Class-Match Matrix (for heatmap) ----
    # For each true class, what fraction of top-5 retrieved are each class?
    k_heatmap = min(5, max_k)
    match_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    for c in range(NUM_CLASSES):
        mask = (test_labels == c)
        if mask.sum() == 0:
            continue
        top_k = retrieved_labels[mask, :k_heatmap]
        for rc in range(NUM_CLASSES):
            match_matrix[c, rc] = float(np.mean(top_k == rc))
    results["class_match_matrix"] = match_matrix.tolist()

    # ---- Store raw data for agreement analysis ----
    results["_retrieved_labels"] = retrieved_labels
    results["_similarities"] = similarities
    results["_test_labels"] = test_labels
    results["_test_sources"] = test_sources

    return results


def compute_agreement_and_rad(results, test_logits, temperature, k_values,
                               alpha=0.5):
    """Compute agreement score and retrieval-augmented accuracy.

    Agreement: does kNN majority class match model prediction?
    RAD accuracy: combine model probs with kNN vote probs.
    """
    retrieved_labels = results["_retrieved_labels"]
    similarities = results["_similarities"]
    test_labels = results["_test_labels"]
    n_test = len(test_labels)

    # Model predictions
    model_probs = torch.softmax(
        torch.from_numpy(test_logits) / temperature, dim=1
    ).numpy()
    model_preds = model_probs.argmax(axis=1)

    agreement_results = {}
    rad_results = {}

    for k in k_values:
        top_k_labels = retrieved_labels[:, :k]
        top_k_sims = similarities[:, :k]

        # ---- kNN predictions (similarity-weighted vote) ----
        knn_probs = np.zeros((n_test, NUM_CLASSES), dtype=np.float64)
        knn_preds = np.zeros(n_test, dtype=np.int64)

        for i in range(n_test):
            for j in range(k):
                lbl = top_k_labels[i, j]
                if 0 <= lbl < NUM_CLASSES:
                    weight = max(0.0, float(top_k_sims[i, j]))
                    knn_probs[i, lbl] += weight
            total = knn_probs[i].sum()
            if total > 0:
                knn_probs[i] /= total
            knn_preds[i] = knn_probs[i].argmax()

        # ---- Agreement ----
        agree_mask = (knn_preds == model_preds)
        agreement = float(np.mean(agree_mask))
        agreement_results[str(k)] = {
            "agreement_rate": agreement,
            "n_agree": int(agree_mask.sum()),
            "n_disagree": int((~agree_mask).sum()),
        }

        # When they agree, accuracy:
        if agree_mask.sum() > 0:
            acc_agree = float(
                np.mean(model_preds[agree_mask] == test_labels[agree_mask])
            )
            agreement_results[str(k)]["accuracy_when_agree"] = acc_agree
        # When they disagree:
        if (~agree_mask).sum() > 0:
            acc_disagree = float(
                np.mean(model_preds[~agree_mask] == test_labels[~agree_mask])
            )
            agreement_results[str(k)]["accuracy_when_disagree"] = acc_disagree

        # ---- RAD: combined prediction ----
        # Method A: weighted average
        combined_probs = alpha * model_probs + (1 - alpha) * knn_probs
        combined_preds = combined_probs.argmax(axis=1)

        # Method B: kNN only
        knn_only_acc = float(np.mean(knn_preds == test_labels))

        # Method C: model only
        model_only_acc = float(np.mean(model_preds == test_labels))

        # Method D: combined
        combined_acc = float(np.mean(combined_preds == test_labels))

        rad_results[str(k)] = {
            "model_only_accuracy": model_only_acc,
            "knn_only_accuracy": knn_only_acc,
            "combined_accuracy": combined_acc,
            "alpha": alpha,
            "improvement_over_model": combined_acc - model_only_acc,
        }

        # Per-class RAD accuracy
        per_class_rad = {}
        for c in range(NUM_CLASSES):
            mask = (test_labels == c)
            if mask.sum() == 0:
                continue
            per_class_rad[CLASS_NAMES[c]] = {
                "model_acc": float(np.mean(model_preds[mask] == c)),
                "knn_acc": float(np.mean(knn_preds[mask] == c)),
                "combined_acc": float(np.mean(combined_preds[mask] == c)),
                "n_samples": int(mask.sum()),
            }
        rad_results[str(k)]["per_class"] = per_class_rad

    results["agreement"] = agreement_results
    results["rad_accuracy"] = rad_results

    # Store for plotting
    results["_model_preds"] = model_preds
    results["_model_probs"] = model_probs

    return results


# ================================================================
# VISUALIZATION
# ================================================================
def plot_recall_at_k(results, output_dir):
    """Plot Recall@K curves overall and per-class."""
    k_values = results["k_values"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall
    ax = axes[0]
    recalls = [results["recall_at_k"][str(k)] for k in k_values]
    precisions = [results["precision_at_k"][str(k)] for k in k_values]
    ax.plot(k_values, recalls, "o-", color="#2196F3", linewidth=2, label="Recall@K")
    ax.plot(k_values, precisions, "s--", color="#FF9800", linewidth=2,
            label="Precision@K")
    for k, r, p in zip(k_values, recalls, precisions):
        ax.annotate(f"{r:.3f}", (k, r), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel("K (number of retrieved cases)")
    ax.set_ylabel("Score")
    ax.set_title("Overall Retrieval Quality")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Per-class Recall@K
    ax = axes[1]
    colors = ["#4CAF50", "#F44336", "#9C27B0", "#FF9800", "#2196F3"]
    for c, (cls_name, cls_data) in enumerate(results["per_class"].items()):
        class_recalls = [cls_data.get(f"recall@{k}", 0) for k in k_values]
        ax.plot(k_values, class_recalls, "o-", color=colors[c % len(colors)],
                linewidth=2, label=cls_name)
    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K")
    ax.set_title("Per-Class Recall@K")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "retrieval_recall_at_k.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_class_match_heatmap(results, output_dir):
    """Plot class-match heatmap: for each true class, distribution of retrieved classes."""
    matrix = np.array(results["class_match_matrix"])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix, annot=True, fmt=".3f", cmap="YlOrRd",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, vmin=0, vmax=1, linewidths=0.5,
        cbar_kws={"label": "Fraction of top-5 retrieved"},
    )
    ax.set_xlabel("Retrieved Class")
    ax.set_ylabel("True Query Class")
    ax.set_title("Class-Match Heatmap (Top-5 Retrieval)")
    plt.tight_layout()
    path = os.path.join(output_dir, "class_match_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_agreement_analysis(results, output_dir):
    """Plot agreement analysis: model vs kNN agreement and accuracy comparison."""
    k_values = results["k_values"]
    agreement = results["agreement"]
    rad = results["rad_accuracy"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Agreement rate by K
    ax = axes[0]
    agree_rates = [agreement[str(k)]["agreement_rate"] for k in k_values]
    ax.bar(range(len(k_values)), agree_rates, color="#2196F3", alpha=0.8)
    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels([f"K={k}" for k in k_values])
    for i, v in enumerate(agree_rates):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("Agreement Rate")
    ax.set_title("Model-Retrieval Agreement")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 2: Accuracy comparison (model vs kNN vs combined)
    ax = axes[1]
    model_acc = [rad[str(k)]["model_only_accuracy"] for k in k_values]
    knn_acc = [rad[str(k)]["knn_only_accuracy"] for k in k_values]
    combined_acc = [rad[str(k)]["combined_accuracy"] for k in k_values]

    x = np.arange(len(k_values))
    width = 0.25
    ax.bar(x - width, model_acc, width, label="Model Only", color="#4CAF50",
           alpha=0.8)
    ax.bar(x, knn_acc, width, label="kNN Only", color="#FF9800", alpha=0.8)
    ax.bar(x + width, combined_acc, width, label="Combined (RAD)",
           color="#2196F3", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in k_values])
    ax.set_ylabel("Accuracy")
    ax.set_title("Retrieval-Augmented Diagnosis Accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 3: Accuracy when agree vs disagree
    ax = axes[2]
    acc_agree = [agreement[str(k)].get("accuracy_when_agree", 0) for k in k_values]
    acc_disagree = [agreement[str(k)].get("accuracy_when_disagree", 0)
                    for k in k_values]
    x = np.arange(len(k_values))
    ax.bar(x - 0.15, acc_agree, 0.3, label="When Agree", color="#4CAF50",
           alpha=0.8)
    ax.bar(x + 0.15, acc_disagree, 0.3, label="When Disagree", color="#F44336",
           alpha=0.8)
    for i, (va, vd) in enumerate(zip(acc_agree, acc_disagree)):
        ax.text(i - 0.15, va + 0.01, f"{va:.3f}", ha="center", fontsize=7)
        ax.text(i + 0.15, vd + 0.01, f"{vd:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in k_values])
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Agreement Status")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "agreement_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ================================================================
# MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="RAD Evaluation: Retrieval-Augmented Diagnosis metrics"
    )
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--k-values", type=int, nargs="+", default=[1, 3, 5, 10],
        help="K values for Recall@K (default: 1 3 5 10)"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Weight for model probs in RAD combination (default: 0.5)"
    )
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model
    if args.model_path:
        model_path = args.model_path
        if not os.path.isabs(model_path):
            model_path = os.path.join(BASE_DIR, model_path)
    else:
        model_path = next(
            (p for p in _MODEL_CANDIDATES if os.path.exists(p)),
            _MODEL_CANDIDATES[0]
        )

    output_dir = args.output_dir or RETRIEVAL_DIR
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(BASE_DIR, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Temperature
    with open(TEMP_PATH) as f:
        temperature = json.load(f)["temperature"]
    print(f"Temperature: {temperature:.4f}")

    # ---- Step 1: Load model ----
    model = load_model(model_path, device)

    # ---- Step 2: Load FAISS index ----
    index, metadata, index_type = load_faiss_index()

    # ---- Step 3: Load test data ----
    test_samples = load_test_samples()
    test_sources = [s["source"] for s in test_samples]

    # ---- Step 4: Extract test embeddings + logits ----
    mean, std = json.load(open(NORM_PATH)).values()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=list(mean), std=list(std)),
    ])

    t0 = time.time()
    test_embeddings, test_logits, test_labels = extract_test_embeddings_and_logits(
        model, test_samples, transform, device,
        batch_size=args.batch_size, use_amp=not args.no_amp,
    )
    print(f"  Embeddings extracted in {time.time()-t0:.1f}s")

    # ---- Step 5: Compute retrieval metrics ----
    results = compute_retrieval_metrics(
        index, metadata, test_embeddings, test_labels,
        test_sources, args.k_values, index_type,
    )

    # ---- Step 6: Compute agreement and RAD accuracy ----
    results = compute_agreement_and_rad(
        results, test_logits, temperature, args.k_values, alpha=args.alpha,
    )

    # ---- Step 7: Generate plots ----
    print("\nGenerating plots...")
    plot_recall_at_k(results, output_dir)
    plot_class_match_heatmap(results, output_dir)
    plot_agreement_analysis(results, output_dir)

    # ---- Step 8: Save results ----
    # Remove non-serializable items
    save_results = {
        k: v for k, v in results.items() if not k.startswith("_")
    }
    results_path = os.path.join(output_dir, "rad_evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"  Saved: {results_path}")

    # ---- Step 9: Print summary ----
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"RAD EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Test samples: {results['n_test']}")
    print(f"  Index vectors: {results['n_index']}")
    print(f"  MAP: {results['map']:.4f}")
    print(f"\n  Recall@K:")
    for k in args.k_values:
        r = results["recall_at_k"][str(k)]
        p = results["precision_at_k"][str(k)]
        print(f"    K={k:2d}: Recall={r:.4f}, Precision={p:.4f}")
    print(f"\n  Per-class AP:")
    for cls_name, cls_data in results["per_class"].items():
        print(f"    {cls_name:15s}: AP={cls_data['ap']:.4f} "
              f"(n={cls_data['n_samples']})")
    print(f"\n  Agreement & RAD (alpha={args.alpha}):")
    for k in args.k_values:
        ag = results["agreement"][str(k)]
        rd = results["rad_accuracy"][str(k)]
        print(f"    K={k:2d}: Agreement={ag['agreement_rate']:.4f}, "
              f"Model={rd['model_only_accuracy']:.4f}, "
              f"kNN={rd['knn_only_accuracy']:.4f}, "
              f"Combined={rd['combined_accuracy']:.4f} "
              f"({'+' if rd['improvement_over_model']>=0 else ''}"
              f"{rd['improvement_over_model']:.4f})")
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
