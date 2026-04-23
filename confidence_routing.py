#!/usr/bin/env python3
"""
confidence_routing.py -- Clinical Confidence Routing for Trustworthy AI
========================================================================
Routes each prediction into one of three tiers based on a triplet of
confidence signals:

  1. Model confidence (softmax probability of predicted class)
  2. Predictive uncertainty (MC Dropout entropy)
  3. Retrieval agreement (does kNN majority match model prediction?)

Routing tiers:
  AUTO-REPORT  : High confidence + Low uncertainty + Retrieval agrees
                 -> Auto-generate clinical screening report
  REVIEW       : Medium confidence or retrieval disagrees
                 -> Flag for specialist review within standard timeframe
  ESCALATE     : Low confidence or high uncertainty or OOD
                 -> Urgent specialist review required

This is a key novelty component for "trustworthy clinical AI" -- the system
knows when to trust its own predictions vs. when to defer to a human expert.

Usage:
    python confidence_routing.py
    python confidence_routing.py --mc-passes 30 --batch-size 32
    python confidence_routing.py --conf-high 0.9 --conf-low 0.4

Outputs:
    outputs_v3/retrieval/confidence_routing_results.json
    outputs_v3/retrieval/routing_distribution.png
    outputs_v3/retrieval/routing_accuracy_by_tier.png
    outputs_v3/retrieval/routing_analysis.png
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
# ROUTING THRESHOLDS (defaults, overridable via CLI)
# ================================================================
DEFAULT_CONF_HIGH = 0.85     # confidence above this -> auto-report candidate
DEFAULT_CONF_LOW = 0.50      # confidence below this -> escalate
DEFAULT_ENTROPY_LOW = 0.5    # entropy below this -> low uncertainty
DEFAULT_ENTROPY_HIGH = 1.0   # entropy above this -> high uncertainty / escalate
DEFAULT_MC_PASSES = 15       # number of MC Dropout forward passes
DEFAULT_RETRIEVAL_K = 5      # k for kNN retrieval agreement


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
    """Load MultiTaskViT from DANN checkpoint."""
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
# DATA
# ================================================================
def resolve_cache_path(csv_cache_path):
    """Resolve cache path with fallback chain."""
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
    samples = []
    with open(TEST_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cache_path = resolve_cache_path(row["cache_path"])
            if cache_path is None:
                continue
            samples.append({
                "image_path": row["image_path"],
                "cache_path": cache_path,
                "label": int(row["disease_label"]),
                "class_name": CLASS_NAMES[int(row["disease_label"])],
                "source": row["source"],
            })
    return samples


def load_faiss_index():
    """Load FAISS index and metadata."""
    ip_path = os.path.join(RETRIEVAL_DIR, "index_flat_ip.faiss")
    l2_path = os.path.join(RETRIEVAL_DIR, "index_flat_l2.faiss")
    meta_path = os.path.join(RETRIEVAL_DIR, "metadata.json")

    if os.path.exists(ip_path):
        index_path, index_type = ip_path, "IP"
    elif os.path.exists(l2_path):
        index_path, index_type = l2_path, "L2"
    else:
        return None, None, None

    index = faiss.read_index(index_path)
    with open(meta_path) as f:
        metadata = json.load(f)
    return index, metadata, index_type


# ================================================================
# CONFIDENCE ROUTING ENGINE
# ================================================================
class ConfidenceRouter:
    """Routes predictions into AUTO-REPORT / REVIEW / ESCALATE tiers.

    Decision logic:
      ESCALATE  if: confidence < conf_low OR entropy > entropy_high
      AUTO-REPORT if: confidence > conf_high AND entropy < entropy_low
                      AND retrieval agrees
      REVIEW    otherwise (the default safe tier)
    """

    TIERS = ["AUTO-REPORT", "REVIEW", "ESCALATE"]

    def __init__(self, conf_high=0.85, conf_low=0.50,
                 entropy_low=0.5, entropy_high=1.0):
        self.conf_high = conf_high
        self.conf_low = conf_low
        self.entropy_low = entropy_low
        self.entropy_high = entropy_high

    def route(self, confidence, entropy, retrieval_agrees):
        """Route a single prediction to a tier.

        Args:
            confidence: float, max softmax probability
            entropy: float, predictive entropy from MC Dropout
            retrieval_agrees: bool, whether kNN majority matches model pred

        Returns:
            str: one of "AUTO-REPORT", "REVIEW", "ESCALATE"
        """
        # ESCALATE: low confidence or high uncertainty
        if confidence < self.conf_low or entropy > self.entropy_high:
            return "ESCALATE"

        # AUTO-REPORT: high confidence + low uncertainty + retrieval agrees
        if (confidence >= self.conf_high
                and entropy < self.entropy_low
                and retrieval_agrees):
            return "AUTO-REPORT"

        # Everything else: REVIEW
        return "REVIEW"

    def route_batch(self, confidences, entropies, retrieval_agrees_flags):
        """Route a batch of predictions."""
        return [
            self.route(c, e, r)
            for c, e, r in zip(confidences, entropies, retrieval_agrees_flags)
        ]


# ================================================================
# MC DROPOUT INFERENCE
# ================================================================
@torch.no_grad()
def mc_dropout_batch(model, batch_tensor, temperature, mc_passes, device):
    """Run MC Dropout for a batch: extract mean probs and entropy.

    Strategy: run backbone once (deterministic), then run disease_head
    T times with dropout enabled.

    Returns:
        mean_probs: (B, C) average softmax probabilities
        entropies: (B,) predictive entropy per sample
    """
    # Run backbone once (deterministic)
    model.eval()
    features = model.backbone(batch_tensor)  # (B, 768)

    # Enable dropout for stochastic passes through the head
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    all_probs = []
    for _ in range(mc_passes):
        f_dropped = model.drop(features)
        logits = model.disease_head(f_dropped)
        probs = torch.softmax(logits / temperature, dim=1)
        all_probs.append(probs.cpu().numpy())

    model.eval()

    all_probs = np.stack(all_probs, axis=0)  # (T, B, C)
    mean_probs = all_probs.mean(axis=0)       # (B, C)

    # Predictive entropy
    entropies = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)  # (B,)

    return mean_probs, entropies


# ================================================================
# RETRIEVAL AGREEMENT
# ================================================================
def compute_retrieval_agreement(model, samples, transform, index, metadata,
                                index_type, device, k=5, batch_size=64,
                                use_amp=True):
    """Compute retrieval agreement for all test samples.

    For each test image:
      1. Extract backbone embedding
      2. Search FAISS for top-K neighbors
      3. Do majority vote among retrieved labels
      4. Check if majority matches model prediction

    Returns:
        knn_preds: (N,) kNN majority class predictions
        knn_probs: (N, C) kNN probability distributions (similarity-weighted)
    """
    n = len(samples)
    meta_labels = np.array([m["label"] for m in metadata])

    all_embeddings = []
    num_batches = (n + batch_size - 1) // batch_size

    amp_enabled = use_amp and device.type == "cuda"
    amp_dtype = torch.float16 if amp_enabled else torch.float32

    model.eval()
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Retrieval embeddings"):
            start = batch_idx * batch_size
            end = min(start + batch_size, n)
            tensors = []
            for s in samples[start:end]:
                img = np.load(s["cache_path"])
                t = transform(img)
                tensors.append(t)
            batch_tensor = torch.stack(tensors).to(device)

            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_enabled):
                features = model.backbone(batch_tensor)
            all_embeddings.append(features.float().cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    faiss.normalize_L2(embeddings)

    # Search
    distances, indices_arr = index.search(embeddings, k)

    knn_probs = np.zeros((n, NUM_CLASSES), dtype=np.float64)
    knn_preds = np.zeros(n, dtype=np.int64)

    for i in range(n):
        for j in range(k):
            idx = indices_arr[i, j]
            if 0 <= idx < len(metadata):
                lbl = meta_labels[idx]
                if index_type == "IP":
                    weight = max(0.0, float(distances[i, j]))
                else:
                    weight = 1.0 / (1.0 + float(distances[i, j]))
                knn_probs[i, lbl] += weight
        total = knn_probs[i].sum()
        if total > 0:
            knn_probs[i] /= total
        knn_preds[i] = knn_probs[i].argmax()

    return knn_preds, knn_probs, embeddings


# ================================================================
# FULL ROUTING EVALUATION
# ================================================================
def evaluate_routing(model, samples, transform, index, metadata, index_type,
                     device, router, temperature, mc_passes=15, k=5,
                     batch_size=64, use_amp=True):
    """Run full routing evaluation on all test samples.

    Steps:
      1. Get model predictions + confidence
      2. Run MC Dropout for uncertainty
      3. Get retrieval agreement
      4. Route each sample
      5. Compute accuracy per tier

    Returns:
        dict with all routing results
    """
    n = len(samples)
    labels = np.array([s["label"] for s in samples])
    sources = [s["source"] for s in samples]

    # ---- Step 1+2: Model predictions + MC Dropout uncertainty ----
    print("Running MC Dropout inference...")
    all_probs = []
    all_entropies = []
    num_batches = (n + batch_size - 1) // batch_size

    amp_enabled = use_amp and device.type == "cuda"
    amp_dtype = torch.float16 if amp_enabled else torch.float32

    for batch_idx in tqdm(range(num_batches), desc="MC Dropout"):
        start = batch_idx * batch_size
        end = min(start + batch_size, n)
        tensors = []
        for s in samples[start:end]:
            img = np.load(s["cache_path"])
            t = transform(img)
            tensors.append(t)
        batch_tensor = torch.stack(tensors).to(device)

        mean_probs, entropies = mc_dropout_batch(
            model, batch_tensor, temperature, mc_passes, device
        )
        all_probs.append(mean_probs)
        all_entropies.append(entropies)

    model_probs = np.concatenate(all_probs, axis=0)      # (N, C)
    entropies = np.concatenate(all_entropies, axis=0)     # (N,)
    model_preds = model_probs.argmax(axis=1)
    confidences = model_probs.max(axis=1)

    # ---- Step 3: Retrieval agreement ----
    print("Computing retrieval agreement...")
    if index is not None:
        knn_preds, knn_probs, _ = compute_retrieval_agreement(
            model, samples, transform, index, metadata, index_type,
            device, k=k, batch_size=batch_size, use_amp=use_amp,
        )
        retrieval_agrees = (knn_preds == model_preds)
    else:
        print("  WARNING: No FAISS index available, retrieval agreement disabled")
        knn_preds = model_preds.copy()
        knn_probs = model_probs.copy()
        retrieval_agrees = np.ones(n, dtype=bool)

    # ---- Step 4: Route each sample ----
    print("Routing predictions...")
    tiers = router.route_batch(confidences, entropies, retrieval_agrees)
    tiers = np.array(tiers)

    # ---- Step 5: Compute metrics per tier ----
    results = {
        "routing_thresholds": {
            "conf_high": router.conf_high,
            "conf_low": router.conf_low,
            "entropy_low": router.entropy_low,
            "entropy_high": router.entropy_high,
        },
        "mc_passes": mc_passes,
        "retrieval_k": k,
        "n_test": n,
    }

    # Overall tier distribution
    tier_counts = Counter(tiers)
    tier_dist = {}
    for tier in ConfidenceRouter.TIERS:
        count = tier_counts.get(tier, 0)
        tier_dist[tier] = {
            "count": count,
            "fraction": count / n if n > 0 else 0.0,
        }
    results["tier_distribution"] = tier_dist

    # Accuracy per tier
    tier_accuracy = {}
    for tier in ConfidenceRouter.TIERS:
        mask = (tiers == tier)
        count = int(mask.sum())
        if count == 0:
            tier_accuracy[tier] = {
                "count": count,
                "accuracy": None,
                "mean_confidence": None,
                "mean_entropy": None,
            }
            continue
        acc = float(np.mean(model_preds[mask] == labels[mask]))
        tier_accuracy[tier] = {
            "count": count,
            "accuracy": acc,
            "mean_confidence": float(np.mean(confidences[mask])),
            "mean_entropy": float(np.mean(entropies[mask])),
            "retrieval_agreement": float(np.mean(retrieval_agrees[mask])),
        }
    results["tier_accuracy"] = tier_accuracy

    # Per-class routing
    per_class_routing = {}
    for c in range(NUM_CLASSES):
        c_mask = (labels == c)
        if c_mask.sum() == 0:
            continue
        c_tiers = tiers[c_mask]
        c_dist = Counter(c_tiers)
        per_class = {"n_samples": int(c_mask.sum())}
        for tier in ConfidenceRouter.TIERS:
            t_mask = (c_tiers == tier)
            cnt = int(t_mask.sum())
            per_class[tier] = {
                "count": cnt,
                "fraction": cnt / c_mask.sum() if c_mask.sum() > 0 else 0.0,
            }
            if cnt > 0:
                # Accuracy among this class routed to this tier
                both_mask = c_mask & (tiers == tier)
                per_class[tier]["accuracy"] = float(
                    np.mean(model_preds[both_mask] == labels[both_mask])
                )
        per_class_routing[CLASS_NAMES[c]] = per_class
    results["per_class_routing"] = per_class_routing

    # Per-source routing
    per_source_routing = {}
    for src in sorted(set(sources)):
        s_mask = np.array([s == src for s in sources])
        if s_mask.sum() == 0:
            continue
        s_tiers = tiers[s_mask]
        per_src = {"n_samples": int(s_mask.sum())}
        for tier in ConfidenceRouter.TIERS:
            t_mask = (s_tiers == tier)
            cnt = int(t_mask.sum())
            per_src[tier] = {
                "count": cnt,
                "fraction": cnt / s_mask.sum() if s_mask.sum() > 0 else 0.0,
            }
        per_source_routing[src] = per_src
    results["per_source_routing"] = per_source_routing

    # Error analysis: where are the errors routed?
    incorrect = (model_preds != labels)
    error_tier_dist = Counter(tiers[incorrect])
    results["error_routing"] = {
        tier: {
            "count": error_tier_dist.get(tier, 0),
            "fraction_of_errors": error_tier_dist.get(tier, 0) / incorrect.sum()
            if incorrect.sum() > 0 else 0.0,
        }
        for tier in ConfidenceRouter.TIERS
    }

    # Safety metric: what % of errors are caught (not auto-reported)?
    auto_errors = int(((tiers == "AUTO-REPORT") & incorrect).sum())
    total_errors = int(incorrect.sum())
    results["safety"] = {
        "total_errors": total_errors,
        "auto_report_errors": auto_errors,
        "error_catch_rate": 1.0 - (auto_errors / total_errors)
        if total_errors > 0 else 1.0,
        "description": "Fraction of errors NOT routed to AUTO-REPORT "
                       "(higher is safer)",
    }

    # Store arrays for plotting
    results["_tiers"] = tiers
    results["_labels"] = labels
    results["_preds"] = model_preds
    results["_confidences"] = confidences
    results["_entropies"] = entropies
    results["_retrieval_agrees"] = retrieval_agrees
    results["_sources"] = sources

    return results


# ================================================================
# VISUALIZATION
# ================================================================
def plot_routing_distribution(results, output_dir):
    """Plot routing tier distribution overall and per-class."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall
    ax = axes[0]
    tiers = ConfidenceRouter.TIERS
    counts = [results["tier_distribution"][t]["count"] for t in tiers]
    fracs = [results["tier_distribution"][t]["fraction"] for t in tiers]
    colors = ["#4CAF50", "#FF9800", "#F44336"]
    bars = ax.bar(tiers, counts, color=colors, alpha=0.85, edgecolor="black",
                  linewidth=0.5)
    for bar, frac in zip(bars, fracs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{frac:.1%}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Number of Predictions")
    ax.set_title("Routing Tier Distribution")
    ax.grid(True, axis="y", alpha=0.3)

    # Per-class
    ax = axes[1]
    x = np.arange(NUM_CLASSES)
    width = 0.25
    for i, tier in enumerate(tiers):
        vals = []
        for c in range(NUM_CLASSES):
            cls = CLASS_NAMES[c]
            if cls in results["per_class_routing"]:
                vals.append(
                    results["per_class_routing"][cls][tier]["fraction"]
                )
            else:
                vals.append(0.0)
        ax.bar(x + i * width - width, vals, width, label=tier,
               color=colors[i], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Fraction")
    ax.set_title("Routing by Disease Class")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "routing_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_routing_accuracy(results, output_dir):
    """Plot accuracy and safety metrics per routing tier."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    tiers = ConfidenceRouter.TIERS
    colors = ["#4CAF50", "#FF9800", "#F44336"]

    # Panel 1: Accuracy per tier
    ax = axes[0]
    accs = []
    for t in tiers:
        a = results["tier_accuracy"][t]["accuracy"]
        accs.append(a if a is not None else 0.0)
    bars = ax.bar(tiers, accs, color=colors, alpha=0.85, edgecolor="black",
                  linewidth=0.5)
    for bar, a in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, a + 0.01,
                f"{a:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy by Routing Tier")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 2: Mean confidence and entropy per tier
    ax = axes[1]
    conf_vals = []
    ent_vals = []
    for t in tiers:
        c = results["tier_accuracy"][t]["mean_confidence"]
        e = results["tier_accuracy"][t]["mean_entropy"]
        conf_vals.append(c if c is not None else 0.0)
        ent_vals.append(e if e is not None else 0.0)
    x = np.arange(len(tiers))
    ax.bar(x - 0.15, conf_vals, 0.3, label="Mean Confidence", color="#2196F3",
           alpha=0.8)
    ax.bar(x + 0.15, ent_vals, 0.3, label="Mean Entropy", color="#9C27B0",
           alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_ylabel("Value")
    ax.set_title("Confidence vs Entropy by Tier")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Panel 3: Error distribution across tiers
    ax = axes[2]
    error_counts = [
        results["error_routing"][t]["count"] for t in tiers
    ]
    total_errors = results["safety"]["total_errors"]
    bars = ax.bar(tiers, error_counts, color=colors, alpha=0.85,
                  edgecolor="black", linewidth=0.5)
    for bar, c in zip(bars, error_counts):
        frac = c / total_errors if total_errors > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2, c + 1,
                f"{c} ({frac:.1%})", ha="center", fontsize=9)
    ax.set_ylabel("Number of Errors")
    ax.set_title(f"Error Distribution (catch rate: "
                 f"{results['safety']['error_catch_rate']:.1%})")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "routing_accuracy_by_tier.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_routing_analysis(results, output_dir):
    """Scatter plot of confidence vs entropy colored by routing tier."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    tiers_arr = results["_tiers"]
    confidences = results["_confidences"]
    entropies = results["_entropies"]
    correct = (results["_preds"] == results["_labels"])

    colors_map = {"AUTO-REPORT": "#4CAF50", "REVIEW": "#FF9800",
                  "ESCALATE": "#F44336"}

    # Panel 1: confidence vs entropy, colored by tier
    ax = axes[0]
    for tier in ConfidenceRouter.TIERS:
        mask = (tiers_arr == tier)
        if mask.sum() == 0:
            continue
        ax.scatter(confidences[mask], entropies[mask],
                   c=colors_map[tier], label=tier, alpha=0.4, s=10)
    ax.axhline(y=results["routing_thresholds"]["entropy_low"], color="gray",
               linestyle="--", alpha=0.5, label=f"Entropy thresholds")
    ax.axhline(y=results["routing_thresholds"]["entropy_high"], color="gray",
               linestyle="--", alpha=0.5)
    ax.axvline(x=results["routing_thresholds"]["conf_high"], color="blue",
               linestyle=":", alpha=0.5, label="Conf thresholds")
    ax.axvline(x=results["routing_thresholds"]["conf_low"], color="blue",
               linestyle=":", alpha=0.5)
    ax.set_xlabel("Model Confidence")
    ax.set_ylabel("Predictive Entropy")
    ax.set_title("Confidence vs Entropy by Routing Tier")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.2)

    # Panel 2: same but colored by correct/incorrect
    ax = axes[1]
    ax.scatter(confidences[correct], entropies[correct],
               c="#4CAF50", label="Correct", alpha=0.3, s=10)
    ax.scatter(confidences[~correct], entropies[~correct],
               c="#F44336", label="Incorrect", alpha=0.5, s=15, marker="x")
    ax.axhline(y=results["routing_thresholds"]["entropy_low"], color="gray",
               linestyle="--", alpha=0.5)
    ax.axhline(y=results["routing_thresholds"]["entropy_high"], color="gray",
               linestyle="--", alpha=0.5)
    ax.axvline(x=results["routing_thresholds"]["conf_high"], color="blue",
               linestyle=":", alpha=0.5)
    ax.axvline(x=results["routing_thresholds"]["conf_low"], color="blue",
               linestyle=":", alpha=0.5)
    ax.set_xlabel("Model Confidence")
    ax.set_ylabel("Predictive Entropy")
    ax.set_title("Confidence vs Entropy: Correct vs Incorrect")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(output_dir, "routing_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ================================================================
# MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Clinical Confidence Routing Evaluation"
    )
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--mc-passes", type=int, default=DEFAULT_MC_PASSES,
                        help=f"MC Dropout passes (default: {DEFAULT_MC_PASSES})")
    parser.add_argument("--retrieval-k", type=int, default=DEFAULT_RETRIEVAL_K,
                        help=f"K for retrieval agreement (default: {DEFAULT_RETRIEVAL_K})")
    parser.add_argument("--conf-high", type=float, default=DEFAULT_CONF_HIGH,
                        help=f"High confidence threshold (default: {DEFAULT_CONF_HIGH})")
    parser.add_argument("--conf-low", type=float, default=DEFAULT_CONF_LOW,
                        help=f"Low confidence threshold (default: {DEFAULT_CONF_LOW})")
    parser.add_argument("--entropy-low", type=float, default=DEFAULT_ENTROPY_LOW,
                        help=f"Low entropy threshold (default: {DEFAULT_ENTROPY_LOW})")
    parser.add_argument("--entropy-high", type=float, default=DEFAULT_ENTROPY_HIGH,
                        help=f"High entropy threshold (default: {DEFAULT_ENTROPY_HIGH})")
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

    # ---- Load resources ----
    model = load_model(model_path, device)
    test_samples = load_test_samples()
    print(f"Test samples: {len(test_samples)}")

    index, metadata, index_type = load_faiss_index()
    if index is not None:
        print(f"FAISS index: {index.ntotal} vectors (type: {index_type})")
    else:
        print("WARNING: No FAISS index found. Retrieval agreement will be disabled.")

    # Transform
    with open(NORM_PATH) as f:
        ns = json.load(f)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=ns["mean_rgb"], std=ns["std_rgb"]),
    ])

    # ---- Build router ----
    router = ConfidenceRouter(
        conf_high=args.conf_high,
        conf_low=args.conf_low,
        entropy_low=args.entropy_low,
        entropy_high=args.entropy_high,
    )
    print(f"Router thresholds: conf=[{args.conf_low}, {args.conf_high}], "
          f"entropy=[{args.entropy_low}, {args.entropy_high}]")

    # ---- Run evaluation ----
    t0 = time.time()
    results = evaluate_routing(
        model, test_samples, transform, index, metadata, index_type,
        device, router, temperature,
        mc_passes=args.mc_passes,
        k=args.retrieval_k,
        batch_size=args.batch_size,
        use_amp=not args.no_amp,
    )
    elapsed = time.time() - t0

    # ---- Generate plots ----
    print("\nGenerating plots...")
    plot_routing_distribution(results, output_dir)
    plot_routing_accuracy(results, output_dir)
    plot_routing_analysis(results, output_dir)

    # ---- Save results ----
    save_results = {k: v for k, v in results.items() if not k.startswith("_")}
    results_path = os.path.join(output_dir, "confidence_routing_results.json")
    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"  Saved: {results_path}")

    # ---- Print summary ----
    print(f"\n{'='*60}")
    print(f"CONFIDENCE ROUTING RESULTS")
    print(f"{'='*60}")
    print(f"  Test samples: {results['n_test']}")
    print(f"  MC Dropout passes: {args.mc_passes}")
    print(f"  Retrieval K: {args.retrieval_k}")
    print(f"\n  Tier Distribution:")
    for tier in ConfidenceRouter.TIERS:
        td = results["tier_distribution"][tier]
        ta = results["tier_accuracy"][tier]
        acc_str = f"{ta['accuracy']:.3f}" if ta["accuracy"] is not None else "N/A"
        print(f"    {tier:12s}: {td['count']:5d} ({td['fraction']:5.1%}) "
              f"| Accuracy: {acc_str}")

    print(f"\n  Safety Metrics:")
    sf = results["safety"]
    print(f"    Total errors: {sf['total_errors']}")
    print(f"    Errors in AUTO-REPORT: {sf['auto_report_errors']}")
    print(f"    Error catch rate: {sf['error_catch_rate']:.1%}")

    print(f"\n  Per-class routing:")
    for cls_name, cls_data in results["per_class_routing"].items():
        parts = []
        for tier in ConfidenceRouter.TIERS:
            parts.append(f"{tier}={cls_data[tier]['fraction']:.1%}")
        print(f"    {cls_name:15s} (n={cls_data['n_samples']:4d}): "
              f"{', '.join(parts)}")

    print(f"\n  Time: {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
