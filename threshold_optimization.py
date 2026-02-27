#!/usr/bin/env python3
"""
Threshold Optimization for RetinaSense v2
==========================================

Optimizes classification thresholds per class to maximize F1 scores.
Current model has AUC=0.91 but uses fixed argmax decision.
With class imbalance, per-class thresholds can significantly improve performance.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}")

# Paths
DATA_DIR = Path('./data')
CACHE_DIR = Path('./preprocessed_cache')
MODEL_PATH = Path('./outputs_v2/best_model.pth')
OUTPUT_DIR = Path('./outputs_v2')
OUTPUT_DIR.mkdir(exist_ok=True)

# Config
BATCH_SIZE = 64
NUM_WORKERS = 8
IMG_SIZE = 300

# Class names
DISEASE_CLASSES = ['Normal', 'Diabetes/DR', 'Glaucoma', 'Cataract', 'AMD']


class CachedDataset(Dataset):
    """Dataset that loads pre-cached preprocessed images"""
    def __init__(self, csv_path, cache_dir, mode='train'):
        self.cache_dir = Path(cache_dir)
        self.mode = mode

        # Load CSV
        df = pd.read_csv(csv_path)

        # Split train/val
        val_size = int(0.15 * len(df))
        if mode == 'train':
            self.df = df.iloc[val_size:].reset_index(drop=True)
        else:
            self.df = df.iloc[:val_size].reset_index(drop=True)

        print(f"📊 {mode.upper()} set: {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['image_id']

        # Load cached image
        cache_path = self.cache_dir / f"{img_id}.npy"
        img = np.load(cache_path)

        # Convert to tensor
        img = torch.from_numpy(img).float()

        # Labels
        disease = int(row['disease_label'])
        severity = int(row['severity_label']) if 'severity_label' in row else 0

        return img, disease, severity, img_id


class MultiTaskModel(nn.Module):
    """Multi-task model for disease classification + severity grading"""
    def __init__(self, num_disease_classes=5, num_severity_classes=5, dropout=0.4):
        super().__init__()

        # Load EfficientNet-B3 backbone
        backbone = models.efficientnet_b3(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Feature dimension
        self.feature_dim = 1536

        # Global pooling and dropout
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)

        # Disease classification head
        self.disease_head = nn.Sequential(
            nn.Linear(1536, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_disease_classes)
        )

        # Severity grading head (simpler than disease head)
        self.severity_head = nn.Sequential(
            nn.Linear(1536, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_severity_classes)
        )

    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        features = self.pool(features)
        features = features.flatten(1)
        features = self.dropout(features)

        # Predictions
        disease_logits = self.disease_head(features)
        severity_logits = self.severity_head(features)

        return disease_logits, severity_logits


def load_model():
    """Load trained model from checkpoint"""
    print(f"📥 Loading model from {MODEL_PATH}")

    model = MultiTaskModel(num_disease_classes=5, num_severity_classes=5, dropout=0.4)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', 'unknown')
    val_acc = checkpoint.get('val_acc', 0)
    val_f1 = checkpoint.get('val_macro_f1', checkpoint.get('val_f1', 0))

    print(f"✅ Loaded model from epoch {epoch}")
    if val_acc > 0:
        print(f"   Val Acc: {val_acc:.2f}%, Macro F1: {val_f1:.3f}")

    return model


def get_predictions(model, dataloader):
    """Get all predictions and ground truth labels"""
    print("🔮 Getting predictions on validation set...")

    all_probs = []
    all_labels = []
    all_ids = []

    with torch.no_grad():
        for imgs, diseases, severities, img_ids in tqdm(dataloader, desc="Predicting"):
            imgs = imgs.to(device, non_blocking=True)

            # Get predictions
            disease_logits, _ = model(imgs)
            probs = torch.softmax(disease_logits, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(diseases.numpy())
            all_ids.extend(img_ids)

    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)

    print(f"✅ Got predictions for {len(all_labels)} samples")
    print(f"   Probability shape: {all_probs.shape}")

    return all_probs, all_labels, all_ids


def find_optimal_threshold_ovr(y_true, y_probs, class_idx):
    """
    Find optimal threshold for one-vs-rest using Youden's J statistic

    Args:
        y_true: Ground truth labels (n_samples,)
        y_probs: Predicted probabilities for this class (n_samples,)
        class_idx: Index of the class

    Returns:
        best_threshold, best_f1
    """
    # Convert to binary (one-vs-rest)
    y_binary = (y_true == class_idx).astype(int)

    # Try thresholds from 0.1 to 0.9
    thresholds = np.arange(0.1, 0.91, 0.01)
    best_f1 = 0
    best_threshold = 0.5

    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)

        # Calculate F1 (handle zero division)
        try:
            f1 = f1_score(y_binary, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
        except:
            continue

    return best_threshold, best_f1


def optimize_thresholds(y_true, y_probs):
    """
    Optimize thresholds for all classes using one-vs-rest approach

    Returns:
        optimal_thresholds: dict mapping class_idx -> threshold
    """
    print("🎯 Optimizing thresholds per class...")

    optimal_thresholds = {}

    for class_idx in range(5):
        class_name = DISEASE_CLASSES[class_idx]
        class_probs = y_probs[:, class_idx]

        # Find optimal threshold
        best_thresh, best_f1 = find_optimal_threshold_ovr(y_true, class_probs, class_idx)

        optimal_thresholds[class_idx] = best_thresh

        # Count samples
        n_samples = (y_true == class_idx).sum()

        print(f"   {class_name:15s}: threshold={best_thresh:.3f}, F1={best_f1:.3f}, n={n_samples}")

    return optimal_thresholds


def predict_with_thresholds(y_probs, thresholds):
    """
    Make predictions using optimized thresholds

    Strategy: For each sample, take the class with highest probability
    if it exceeds its threshold. Otherwise, predict the most likely class.
    """
    n_samples = y_probs.shape[0]
    predictions = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        probs = y_probs[i]

        # Get class with max probability
        max_class = np.argmax(probs)
        max_prob = probs[max_class]

        # Check if it exceeds threshold
        if max_prob >= thresholds[max_class]:
            predictions[i] = max_class
        else:
            # Try other classes in order of probability
            sorted_classes = np.argsort(probs)[::-1]
            assigned = False
            for cls in sorted_classes:
                if probs[cls] >= thresholds[cls]:
                    predictions[i] = cls
                    assigned = True
                    break

            # If no class exceeds threshold, fall back to max probability
            if not assigned:
                predictions[i] = max_class

    return predictions


def evaluate(y_true, y_pred, y_probs, title="Evaluation"):
    """Comprehensive evaluation with all metrics"""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")

    # Overall metrics
    accuracy = (y_true == y_pred).mean() * 100
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Macro F1: {macro_f1:.3f}")
    print(f"Weighted F1: {weighted_f1:.3f}")

    # AUC-ROC
    try:
        auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
        print(f"Macro AUC-ROC: {auc:.3f}")
    except:
        auc = 0.0
        print("AUC-ROC: N/A")

    # Per-class metrics
    print(f"\n{'Class':<15} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Supp':>6}")
    print("-" * 50)

    f1_scores = f1_score(y_true, y_pred, average=None, zero_division=0)
    precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)

    per_class_results = {}

    for i, class_name in enumerate(DISEASE_CLASSES):
        support = (y_true == i).sum()
        per_class_results[class_name] = {
            'f1': f1_scores[i],
            'precision': precisions[i],
            'recall': recalls[i],
            'support': int(support)
        }
        print(f"{class_name:<15} {f1_scores[i]:>6.3f} {precisions[i]:>6.3f} {recalls[i]:>6.3f} {support:>6d}")

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'auc': auc,
        'per_class': per_class_results,
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }


def plot_comparison(results_baseline, results_optimized, optimal_thresholds, output_path):
    """Plot before/after comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # F1 scores comparison
    ax = axes[0, 0]
    classes = DISEASE_CLASSES
    baseline_f1 = [results_baseline['per_class'][c]['f1'] for c in classes]
    optimized_f1 = [results_optimized['per_class'][c]['f1'] for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    ax.bar(x - width/2, baseline_f1, width, label='Baseline (argmax)', alpha=0.8)
    ax.bar(x + width/2, optimized_f1, width, label='Optimized thresholds', alpha=0.8)

    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Class F1 Score Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Overall metrics comparison
    ax = axes[0, 1]
    metrics = ['Accuracy', 'Macro F1', 'Weighted F1', 'AUC-ROC']
    baseline_vals = [
        results_baseline['accuracy']/100,
        results_baseline['macro_f1'],
        results_baseline['weighted_f1'],
        results_baseline['auc']
    ]
    optimized_vals = [
        results_optimized['accuracy']/100,
        results_optimized['macro_f1'],
        results_optimized['weighted_f1'],
        results_optimized['auc']
    ]

    x = np.arange(len(metrics))
    ax.bar(x - width/2, baseline_vals, width, label='Baseline', alpha=0.8)
    ax.bar(x + width/2, optimized_vals, width, label='Optimized', alpha=0.8)

    ax.set_ylabel('Score')
    ax.set_title('Overall Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    # Optimal thresholds
    ax = axes[1, 0]
    thresholds_list = [optimal_thresholds[i] for i in range(5)]
    bars = ax.bar(classes, thresholds_list, alpha=0.8, color='steelblue')

    # Add default threshold line
    ax.axhline(y=0.5, color='red', linestyle='--', label='Default (0.5)', alpha=0.5)

    ax.set_ylabel('Optimal Threshold')
    ax.set_title('Optimized Thresholds per Class')
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    # Add threshold values on bars
    for bar, thresh in zip(bars, thresholds_list):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{thresh:.2f}',
                ha='center', va='bottom', fontsize=9)

    # Improvement heatmap
    ax = axes[1, 1]
    improvements = []
    for class_name in classes:
        baseline = results_baseline['per_class'][class_name]['f1']
        optimized = results_optimized['per_class'][class_name]['f1']
        improvement = optimized - baseline
        improvements.append(improvement)

    colors = ['red' if x < 0 else 'green' for x in improvements]
    bars = ax.barh(classes, improvements, color=colors, alpha=0.7)

    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('F1 Score Change')
    ax.set_title('Per-Class F1 Improvement')
    ax.grid(axis='x', alpha=0.3)

    # Add values
    for i, (bar, val) in enumerate(zip(bars, improvements)):
        x_pos = val + (0.01 if val > 0 else -0.01)
        ha = 'left' if val > 0 else 'right'
        ax.text(x_pos, i, f'{val:+.3f}', va='center', ha=ha, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Comparison plot saved to {output_path}")


def main():
    print("🎯 Threshold Optimization for RetinaSense v2")
    print("=" * 50)

    # Load model
    model = load_model()

    # Load validation data
    val_dataset = CachedDataset(
        csv_path=DATA_DIR / 'train_processed.csv',
        cache_dir=CACHE_DIR,
        mode='val'
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )

    # Get predictions
    y_probs, y_true, img_ids = get_predictions(model, val_loader)

    # Baseline: argmax predictions
    y_pred_baseline = np.argmax(y_probs, axis=1)

    # Evaluate baseline
    print("\n" + "="*50)
    print("BASELINE EVALUATION (argmax)")
    print("="*50)
    results_baseline = evaluate(y_true, y_pred_baseline, y_probs, "Baseline")

    # Optimize thresholds
    print("\n" + "="*50)
    print("THRESHOLD OPTIMIZATION")
    print("="*50)
    optimal_thresholds = optimize_thresholds(y_true, y_probs)

    # Predict with optimized thresholds
    y_pred_optimized = predict_with_thresholds(y_probs, optimal_thresholds)

    # Evaluate optimized
    print("\n" + "="*50)
    print("OPTIMIZED EVALUATION")
    print("="*50)
    results_optimized = evaluate(y_true, y_pred_optimized, y_probs, "Optimized")

    # Save results
    results = {
        'optimal_thresholds': optimal_thresholds,
        'baseline': results_baseline,
        'optimized': results_optimized
    }

    output_json = OUTPUT_DIR / 'threshold_optimization_results.json'
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {output_json}")

    # Plot comparison
    plot_path = OUTPUT_DIR / 'threshold_comparison.png'
    plot_comparison(results_baseline, results_optimized, optimal_thresholds, plot_path)

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Baseline Macro F1:   {results_baseline['macro_f1']:.3f}")
    print(f"Optimized Macro F1:  {results_optimized['macro_f1']:.3f}")
    print(f"Improvement:         {results_optimized['macro_f1'] - results_baseline['macro_f1']:+.3f}")
    print(f"\nBaseline Accuracy:   {results_baseline['accuracy']:.2f}%")
    print(f"Optimized Accuracy:  {results_optimized['accuracy']:.2f}%")
    print(f"Improvement:         {results_optimized['accuracy'] - results_baseline['accuracy']:+.2f}%")

    print("\n✅ Threshold optimization complete!")
    print(f"📁 Results saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
