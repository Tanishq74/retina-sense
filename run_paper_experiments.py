#!/usr/bin/env python3
"""
run_paper_experiments.py -- Run all experiments for the IEEE paper.
=================================================================

Master orchestration script for RetinaSense-ViT IEEE paper experiments.
Each experiment is independently skippable via CLI flags.

Experiments:
  1. Rebuild FAISS index           (--rebuild-faiss)  ~5-10 min
  2. RAD evaluation                (--eval-rad)       ~10 min
  3. Leave-One-Domain-Out (LODO)   (--lodo)           ~2 hrs
  4. Ablation study                (--ablation)        ~1.5 hrs
  5. Confidence routing eval       (--eval-routing)   ~5 min

Usage:
    python run_paper_experiments.py --all               # Run everything (~3-4 hrs)
    python run_paper_experiments.py --rebuild-faiss      # Just rebuild FAISS
    python run_paper_experiments.py --lodo               # Just leave-one-domain-out
    python run_paper_experiments.py --ablation           # Just ablation study
    python run_paper_experiments.py --lodo --ablation    # Both heavy experiments
"""

import os
import sys
import time
import json
import copy
import math
import argparse
import subprocess
import warnings
import random
import traceback
import numpy as np
import pandas as pd
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from tqdm import tqdm
from collections import Counter, defaultdict
from datetime import datetime

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision import transforms

import timm

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score,
)


# ================================================================
# IMPORT FROM train_dann_v3.py
# ================================================================
# We import the core building blocks directly so we use the exact
# same architecture, dataset class, and utilities as production.
from train_dann_v3 import (
    # Architecture
    DANNMultiTaskViT,
    GRL,
    GradientReversalFunction,
    FocalLoss,
    # Data
    RetinalDANNv3Dataset,
    make_transforms,
    load_norm_stats,
    resolve_cache_path,
    preprocess_image,
    # Training utilities
    get_optimizer_with_llrd,
    ganin_lambda,
    compute_ece,
    set_seed,
    # Evaluation
    evaluate,
    evaluate_domain,
    collect_logits_labels,
    # Config
    Config as DANNConfig,
)


# ================================================================
# CONSTANTS
# ================================================================
CLASS_NAMES = ['Normal', 'Diabetes/DR', 'Glaucoma', 'Cataract', 'AMD']
NUM_CLASSES = 5
SEED = 42

DOMAIN_MAP_FULL = {
    'APTOS': 0,
    'ODIR': 1,
    'REFUGE2': 2,
    'MESSIDOR2': 3,
}

PRODUCTION_CKPT = './outputs_v3/dann_v3/best_model.pth'
OUTPUT_DIR = './outputs_v3'

# CSV paths (prefer expanded, fall back to original)
TRAIN_CSV = './data/train_split_expanded.csv'
TRAIN_CSV_FALLBACK = './data/train_split.csv'
CALIB_CSV = './data/calib_split_expanded.csv'
CALIB_CSV_FALLBACK = './data/calib_split.csv'
TEST_CSV = './data/test_split.csv'


# ================================================================
# CLI ARGUMENTS
# ================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description='RetinaSense IEEE Paper Experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--all', action='store_true',
                   help='Run all experiments')
    p.add_argument('--rebuild-faiss', action='store_true',
                   help='Experiment 1: Rebuild FAISS index')
    p.add_argument('--eval-rad', action='store_true',
                   help='Experiment 2: RAD evaluation')
    p.add_argument('--lodo', action='store_true',
                   help='Experiment 3: Leave-One-Domain-Out validation')
    p.add_argument('--ablation', action='store_true',
                   help='Experiment 4: Ablation study')
    p.add_argument('--eval-routing', action='store_true',
                   help='Experiment 5: Confidence routing evaluation')
    # Shared training params
    p.add_argument('--epochs', type=int, default=20,
                   help='Training epochs for LODO/ablation (default: 20)')
    p.add_argument('--batch-size', type=int, default=32,
                   help='Batch size (default: 32)')
    p.add_argument('--workers', type=int, default=8,
                   help='DataLoader workers (default: 8)')
    p.add_argument('--lr', type=float, default=3e-5,
                   help='Base learning rate (default: 3e-5)')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed (default: 42)')
    p.add_argument('--gpu', type=int, default=0,
                   help='GPU device index (default: 0)')
    return p.parse_args()


# ================================================================
# UTILITY FUNCTIONS
# ================================================================
def log_section(title, char='=', width=75):
    """Print a formatted section header."""
    print(f'\n{char * width}')
    print(f'  {title}')
    print(f'{char * width}')


def log_subsection(title, char='-', width=60):
    """Print a formatted subsection header."""
    print(f'\n  {char * width}')
    print(f'  {title}')
    print(f'  {char * width}')


def format_duration(seconds):
    """Format seconds as human-readable duration."""
    if seconds < 60:
        return f'{seconds:.0f}s'
    elif seconds < 3600:
        return f'{seconds / 60:.1f}m'
    else:
        return f'{seconds / 3600:.1f}h'


def load_data_splits():
    """Load train, calib, test DataFrames with fallback chain."""
    # Train
    if os.path.exists(TRAIN_CSV):
        train_df = pd.read_csv(TRAIN_CSV)
        print(f'    Train: {TRAIN_CSV} ({len(train_df)} samples)')
    elif os.path.exists(TRAIN_CSV_FALLBACK):
        train_df = pd.read_csv(TRAIN_CSV_FALLBACK)
        print(f'    Train: {TRAIN_CSV_FALLBACK} ({len(train_df)} samples) [FALLBACK]')
    else:
        raise FileNotFoundError(
            f'No train CSV found at {TRAIN_CSV} or {TRAIN_CSV_FALLBACK}'
        )

    # Calib
    if os.path.exists(CALIB_CSV):
        calib_df = pd.read_csv(CALIB_CSV)
        print(f'    Calib: {CALIB_CSV} ({len(calib_df)} samples)')
    elif os.path.exists(CALIB_CSV_FALLBACK):
        calib_df = pd.read_csv(CALIB_CSV_FALLBACK)
        print(f'    Calib: {CALIB_CSV_FALLBACK} ({len(calib_df)} samples) [FALLBACK]')
    else:
        raise FileNotFoundError(
            f'No calib CSV found at {CALIB_CSV} or {CALIB_CSV_FALLBACK}'
        )

    # Test
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f'Test CSV not found: {TEST_CSV}')
    test_df = pd.read_csv(TEST_CSV)
    print(f'    Test:  {TEST_CSV} ({len(test_df)} samples)')

    return train_df, calib_df, test_df


def make_weighted_sampler(labels, n_classes=5):
    """Create a WeightedRandomSampler from label array."""
    class_cnt = np.bincount(labels, minlength=n_classes).astype(float)
    class_cnt = np.where(class_cnt == 0, 1.0, class_cnt)
    weights = 1.0 / class_cnt[labels]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True,
    )


def compute_full_metrics(preds, targets, probs, class_names=CLASS_NAMES):
    """Compute comprehensive metrics dict."""
    acc = 100.0 * (preds == targets).mean()
    mf1 = f1_score(targets, preds, average='macro', zero_division=0)
    wf1 = f1_score(targets, preds, average='weighted', zero_division=0)
    per_f1 = f1_score(
        targets, preds, average=None,
        labels=range(len(class_names)), zero_division=0,
    )
    try:
        mauc = roc_auc_score(
            targets, probs, multi_class='ovr', average='macro',
        )
    except Exception:
        mauc = 0.0

    ece = compute_ece(probs, targets)

    return {
        'accuracy': float(acc),
        'macro_f1': float(mf1),
        'weighted_f1': float(wf1),
        'macro_auc': float(mauc),
        'ece': float(ece),
        **{f'f1_{class_names[i]}': float(per_f1[i])
           for i in range(len(class_names))},
    }


def print_metrics_table(metrics, label=''):
    """Print a formatted metrics summary."""
    if label:
        print(f'\n    [{label}]')
    print(f'    Accuracy   : {metrics["accuracy"]:.2f}%')
    print(f'    Macro F1   : {metrics["macro_f1"]:.4f}')
    print(f'    Weighted F1: {metrics["weighted_f1"]:.4f}')
    print(f'    Macro AUC  : {metrics["macro_auc"]:.4f}')
    print(f'    ECE        : {metrics["ece"]:.4f}')
    for cn in CLASS_NAMES:
        key = f'f1_{cn}'
        if key in metrics:
            print(f'      {cn:15s}: F1={metrics[key]:.3f}')


def run_subprocess(script_name, desc):
    """Run an external script via subprocess with error handling."""
    script_path = os.path.join('.', script_name)
    if not os.path.exists(script_path):
        print(f'    WARNING: {script_path} not found.')
        print(f'    This script is being written by another agent.')
        print(f'    Skipping {desc}.')
        return False

    print(f'    Running: python {script_path}')
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=False,
            timeout=3600,  # 1 hour timeout
        )
        if result.returncode != 0:
            print(f'    WARNING: {script_name} exited with code {result.returncode}')
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f'    WARNING: {script_name} timed out after 1 hour')
        return False
    except Exception as e:
        print(f'    WARNING: Failed to run {script_name}: {e}')
        return False


# ================================================================
# EXPERIMENT 1: REBUILD FAISS INDEX
# ================================================================
def experiment_rebuild_faiss():
    """Rebuild the FAISS similarity retrieval index."""
    log_section('EXPERIMENT 1: Rebuild FAISS Index')
    t0 = time.time()

    success = run_subprocess('rebuild_faiss_full.py', 'FAISS rebuild')
    elapsed = time.time() - t0

    if success:
        print(f'\n    FAISS rebuild completed in {format_duration(elapsed)}')
    return success


# ================================================================
# EXPERIMENT 2: RAD EVALUATION
# ================================================================
def experiment_eval_rad():
    """Run RAD (Retrieval-Augmented Diagnosis) evaluation."""
    log_section('EXPERIMENT 2: RAD Evaluation')
    t0 = time.time()

    success = run_subprocess('rad_evaluation.py', 'RAD evaluation')
    elapsed = time.time() - t0

    if success:
        print(f'\n    RAD evaluation completed in {format_duration(elapsed)}')
    return success


# ================================================================
# EXPERIMENT 3: LEAVE-ONE-DOMAIN-OUT (LODO) VALIDATION
# ================================================================
def train_dann_for_lodo(train_df, val_df, domain_map, num_domains,
                        device, norm_mean, norm_std, args,
                        output_prefix='lodo', desc='LODO'):
    """
    Train a DANN model on a given train/val split.

    This is a simplified training loop matching the DANN-v3 architecture
    but without hard-mining or mixup (clean ablation).

    Args:
        train_df: DataFrame for training
        val_df: DataFrame for validation/testing
        domain_map: dict mapping source name -> domain index
        num_domains: number of domain classes
        device: torch device
        norm_mean, norm_std: normalisation stats
        args: CLI args (epochs, lr, batch_size, etc.)
        output_prefix: name prefix for logging
        desc: description string

    Returns:
        dict with metrics on val_df
    """
    set_seed(args.seed)

    # Build transforms
    train_transform = make_transforms('train', norm_mean, norm_std)
    val_transform = make_transforms('val', norm_mean, norm_std)

    # Build datasets
    train_ds = RetinalDANNv3Dataset(train_df, train_transform, domain_map)
    val_ds = RetinalDANNv3Dataset(val_df, val_transform, domain_map)

    # Sampler with class balancing
    sampler = make_weighted_sampler(
        train_df['disease_label'].values, n_classes=NUM_CLASSES,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    # Build model
    model = DANNMultiTaskViT(
        n_disease=NUM_CLASSES,
        n_severity=5,
        num_domains=num_domains,
        drop=0.3,
        backbone_name='vit_base_patch16_224',
    ).to(device)

    # Compute class weights from this specific train split
    # Handle missing classes (some domains don't have all 5 classes)
    present_labels = train_df['disease_label'].unique()
    cw = np.ones(NUM_CLASSES, dtype=np.float32)
    if len(present_labels) < NUM_CLASSES:
        present_cw = compute_class_weight(
            'balanced',
            classes=np.sort(present_labels),
            y=train_df['disease_label'].values,
        )
        for i, lbl in enumerate(np.sort(present_labels)):
            cw[lbl] = present_cw[i]
    else:
        cw = compute_class_weight(
            'balanced',
            classes=np.arange(NUM_CLASSES),
            y=train_df['disease_label'].values,
        )
    alpha = torch.tensor(cw, dtype=torch.float32).to(device)
    alpha = alpha / alpha.sum() * NUM_CLASSES

    # Loss functions
    criterion_d = FocalLoss(
        alpha=alpha, gamma=2.0, label_smoothing=0.1,
    )
    criterion_s = nn.CrossEntropyLoss(
        ignore_index=-1, label_smoothing=0.1,
    )
    criterion_dom = nn.CrossEntropyLoss()

    # Optimizer with LLRD
    optimizer = get_optimizer_with_llrd(
        model, base_lr=args.lr, decay_factor=0.85, weight_decay=1e-4,
    )

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    scaler = GradScaler()
    grad_accum = 2  # effective batch = 64

    # Training loop
    best_f1 = 0.0
    best_state = None
    patience_ctr = 0
    patience = 10

    for epoch in range(args.epochs):
        t0 = time.time()
        lam_p = ganin_lambda(epoch, args.epochs, max_lambda=0.3)

        # Train one epoch
        model.train()
        run_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            imgs = batch[0].to(device, non_blocking=True)
            d_lbl = batch[1].to(device, non_blocking=True)
            s_lbl = batch[2].to(device, non_blocking=True)
            dom_lbl = batch[3].to(device, non_blocking=True)

            with autocast('cuda'):
                d_out, s_out, dom_out = model(imgs, alpha=lam_p)
                loss_d = criterion_d(d_out, d_lbl)
                loss_s = criterion_s(s_out, s_lbl)
                loss_dom = criterion_dom(dom_out, dom_lbl)
                loss = (loss_d + 0.2 * loss_s +
                        0.05 * lam_p * loss_dom) / grad_accum

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            run_loss += loss.item() * grad_accum
            with torch.no_grad():
                preds = d_out.argmax(1)
                correct += (preds == d_lbl).sum().item()
                total += d_lbl.size(0)

        # Flush remaining gradients
        if len(train_loader) % grad_accum != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        train_loss = run_loss / max(len(train_loader), 1)
        train_acc = 100.0 * correct / max(total, 1)

        # Validate
        val_loss, val_preds, val_targets, val_probs = evaluate(
            val_loader, model, criterion_d, criterion_s, device,
            desc=f'{desc} E{epoch+1}',
        )
        val_acc = 100.0 * (val_preds == val_targets).mean()
        mf1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)

        elapsed = time.time() - t0
        tag = ''
        if mf1 > best_f1 + 0.001:
            best_f1 = mf1
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
            tag = ' *'
        else:
            patience_ctr += 1

        if (epoch + 1) % 5 == 0 or epoch == 0 or tag:
            print(f'      E{epoch+1:3d}/{args.epochs} | '
                  f'TrL={train_loss:.3f} TrA={train_acc:.1f}% | '
                  f'VL={val_loss:.3f} VA={val_acc:.1f}% | '
                  f'mF1={mf1:.4f} | {elapsed:.0f}s{tag}')

        if patience_ctr >= patience:
            print(f'      Early stopped at epoch {epoch+1} (patience={patience})')
            break

    # Load best model and compute final metrics
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    _, final_preds, final_targets, final_probs = evaluate(
        val_loader, model, criterion_d, criterion_s, device,
        desc=f'{desc} final',
    )

    metrics = compute_full_metrics(final_preds, final_targets, final_probs)
    metrics['best_f1'] = float(best_f1)
    metrics['epochs_trained'] = epoch + 1

    # Clean up model to free GPU memory
    del model, optimizer, scaler
    torch.cuda.empty_cache()

    return metrics


def experiment_lodo(args, device):
    """
    Leave-One-Domain-Out (LODO) cross-validation.

    For each of the 4 domains (APTOS, ODIR, REFUGE2, MESSIDOR2):
    - Hold out that domain entirely from training
    - Train DANN-v3 on the remaining 3 domains
    - Test on the held-out domain
    - Report accuracy, F1, AUC
    """
    log_section('EXPERIMENT 3: Leave-One-Domain-Out (LODO) Validation')
    t0_total = time.time()

    print('\n  Loading data...')
    train_df, calib_df, test_df = load_data_splits()

    # Combine train + calib for LODO (test stays sealed)
    all_train_df = pd.concat([train_df, calib_df], ignore_index=True)
    print(f'    Combined train+calib: {len(all_train_df)} samples')

    # Combine all available data including test for the held-out domain eval
    all_data_df = pd.concat([train_df, calib_df, test_df], ignore_index=True)
    print(f'    Total dataset: {len(all_data_df)} samples')

    # Normalisation stats
    norm_mean, norm_std = load_norm_stats()

    # Get all unique domains
    all_data_df['source_upper'] = all_data_df['source'].str.upper()
    domains = sorted(all_data_df['source_upper'].unique())
    print(f'    Domains found: {domains}')

    lodo_results = {}
    summary_rows = []

    for held_out_domain in domains:
        log_subsection(f'LODO: Held-out domain = {held_out_domain}')
        t0_fold = time.time()

        # Split: train on everything EXCEPT held-out domain
        mask_heldout = all_data_df['source_upper'] == held_out_domain
        lodo_train_df = all_data_df[~mask_heldout].copy().reset_index(drop=True)
        lodo_test_df = all_data_df[mask_heldout].copy().reset_index(drop=True)

        n_train = len(lodo_train_df)
        n_test = len(lodo_test_df)

        print(f'    Train: {n_train} samples (domains: '
              f'{sorted(lodo_train_df["source_upper"].unique().tolist())})')
        print(f'    Test:  {n_test} samples (domain: {held_out_domain})')

        if n_test == 0:
            print(f'    WARNING: No test samples for {held_out_domain}, skipping.')
            continue

        # Print class distribution in the held-out domain
        test_labels = lodo_test_df['disease_label'].values
        label_counts = np.bincount(test_labels, minlength=NUM_CLASSES)
        dist_str = ', '.join(
            f'{CLASS_NAMES[i]}={label_counts[i]}'
            for i in range(NUM_CLASSES)
        )
        print(f'    Test class distribution: {dist_str}')

        # Build domain map for the training domains only
        train_domains = sorted(lodo_train_df['source_upper'].unique().tolist())
        lodo_domain_map = {d: i for i, d in enumerate(train_domains)}
        # Add the held-out domain with a dummy index (won't be used in training)
        if held_out_domain not in lodo_domain_map:
            lodo_domain_map[held_out_domain] = len(lodo_domain_map)
        num_domains_lodo = len(train_domains)  # only training domains for DANN

        print(f'    DANN domains ({num_domains_lodo}): {lodo_domain_map}')

        # Train
        metrics = train_dann_for_lodo(
            train_df=lodo_train_df,
            val_df=lodo_test_df,
            domain_map=lodo_domain_map,
            num_domains=len(lodo_domain_map),
            device=device,
            norm_mean=norm_mean,
            norm_std=norm_std,
            args=args,
            desc=f'LODO-{held_out_domain}',
        )

        elapsed_fold = time.time() - t0_fold

        # Store results
        metrics['held_out_domain'] = held_out_domain
        metrics['n_train'] = n_train
        metrics['n_test'] = n_test
        metrics['time_seconds'] = float(elapsed_fold)
        lodo_results[held_out_domain] = metrics

        print_metrics_table(metrics, label=f'LODO held-out={held_out_domain}')
        print(f'    Time: {format_duration(elapsed_fold)}')

        summary_rows.append({
            'domain': held_out_domain,
            'n_test': n_test,
            'accuracy': metrics['accuracy'],
            'macro_f1': metrics['macro_f1'],
            'macro_auc': metrics['macro_auc'],
        })

    # Summary table
    log_subsection('LODO Summary')
    print(f'\n    {"Domain":12s} | {"N_test":>6s} | {"Acc(%)":>7s} | '
          f'{"M-F1":>6s} | {"M-AUC":>6s}')
    print(f'    {"-"*12}-+-{"-"*6}-+-{"-"*7}-+-{"-"*6}-+-{"-"*6}')
    for row in summary_rows:
        print(f'    {row["domain"]:12s} | {row["n_test"]:6d} | '
              f'{row["accuracy"]:7.2f} | {row["macro_f1"]:.4f} | '
              f'{row["macro_auc"]:.4f}')

    if summary_rows:
        avg_acc = np.mean([r['accuracy'] for r in summary_rows])
        avg_f1 = np.mean([r['macro_f1'] for r in summary_rows])
        avg_auc = np.mean([r['macro_auc'] for r in summary_rows])
        print(f'    {"AVERAGE":12s} | {"":>6s} | {avg_acc:7.2f} | '
              f'{avg_f1:.4f} | {avg_auc:.4f}')

        lodo_results['_summary'] = {
            'avg_accuracy': float(avg_acc),
            'avg_macro_f1': float(avg_f1),
            'avg_macro_auc': float(avg_auc),
            'domains_tested': [r['domain'] for r in summary_rows],
        }

    # Save results
    results_path = os.path.join(OUTPUT_DIR, 'lodo_results.json')
    with open(results_path, 'w') as f:
        json.dump(lodo_results, f, indent=2)
    print(f'\n    Results saved to {results_path}')

    # Generate bar chart
    _plot_lodo_chart(summary_rows)

    total_elapsed = time.time() - t0_total
    print(f'\n    Total LODO time: {format_duration(total_elapsed)}')

    return lodo_results


def _plot_lodo_chart(summary_rows):
    """Generate LODO results bar chart."""
    if not summary_rows:
        return

    domains = [r['domain'] for r in summary_rows]
    accs = [r['accuracy'] for r in summary_rows]
    f1s = [r['macro_f1'] for r in summary_rows]
    aucs = [r['macro_auc'] for r in summary_rows]

    x = np.arange(len(domains))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width, accs, width, label='Accuracy (%)',
                   color='#2ecc71', alpha=0.85)
    bars2 = ax.bar(x, [f * 100 for f in f1s], width, label='Macro F1 (x100)',
                   color='#3498db', alpha=0.85)
    bars3 = ax.bar(x + width, [a * 100 for a in aucs], width,
                   label='Macro AUC (x100)', color='#e74c3c', alpha=0.85)

    ax.set_xlabel('Held-Out Domain', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Leave-One-Domain-Out (LODO) Validation Results',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(domains, fontsize=11)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    # Value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, 'lodo_chart.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    LODO chart saved to {chart_path}')


# ================================================================
# EXPERIMENT 4: ABLATION STUDY
# ================================================================
class MultiTaskViT(nn.Module):
    """
    Base ViT without DANN -- for ablation variant 1 (no domain adaptation).
    Same architecture as DANNMultiTaskViT minus the domain head and GRL.
    """

    def __init__(self, n_disease=5, n_severity=5, drop=0.3,
                 backbone_name='vit_base_patch16_224'):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0,
        )
        feat = self.backbone.num_features

        self.drop = nn.Dropout(drop)

        self.disease_head = nn.Sequential(
            nn.Linear(feat, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_disease),
        )

        self.severity_head = nn.Sequential(
            nn.Linear(feat, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_severity),
        )

    def forward(self, x):
        f = self.backbone(x)
        f = self.drop(f)
        return self.disease_head(f), self.severity_head(f)

    def forward_no_domain(self, x):
        """Compatibility method so evaluate() works."""
        return self.forward(x)


def train_base_vit(train_df, val_df, device, norm_mean, norm_std, args,
                   desc='BaseViT'):
    """
    Train a base ViT (no DANN, no improvements) for ablation.
    Uses standard CE loss, no label smoothing, no mixup, no hard mining.
    """
    set_seed(args.seed)

    train_transform = make_transforms('train', norm_mean, norm_std)
    val_transform = make_transforms('val', norm_mean, norm_std)

    # Use a simple dataset wrapper that works with evaluate()
    # RetinalDANNv3Dataset returns (img, disease, severity, domain, idx)
    # evaluate() uses batch[0], batch[1], batch[2] -- so this works.
    dummy_domain_map = {'APTOS': 0, 'ODIR': 1, 'REFUGE2': 2, 'MESSIDOR2': 3}
    train_ds = RetinalDANNv3Dataset(train_df, train_transform, dummy_domain_map)
    val_ds = RetinalDANNv3Dataset(val_df, val_transform, dummy_domain_map)

    sampler = make_weighted_sampler(
        train_df['disease_label'].values, n_classes=NUM_CLASSES,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    model = MultiTaskViT(
        n_disease=NUM_CLASSES, n_severity=5, drop=0.3,
    ).to(device)

    # Standard CE loss (no focal loss, no label smoothing)
    cw = compute_class_weight(
        'balanced',
        classes=np.arange(NUM_CLASSES),
        y=train_df['disease_label'].values,
    )
    weight_tensor = torch.tensor(cw, dtype=torch.float32).to(device)
    criterion_d = nn.CrossEntropyLoss(weight=weight_tensor)
    criterion_s = nn.CrossEntropyLoss(ignore_index=-1)

    # Standard AdamW (no LLRD since this is the "base" variant)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    scaler = GradScaler()
    grad_accum = 2

    best_f1 = 0.0
    best_state = None
    patience_ctr = 0

    for epoch in range(args.epochs):
        model.train()
        run_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            imgs = batch[0].to(device, non_blocking=True)
            d_lbl = batch[1].to(device, non_blocking=True)
            s_lbl = batch[2].to(device, non_blocking=True)

            with autocast('cuda'):
                d_out, s_out = model(imgs)
                loss_d = criterion_d(d_out, d_lbl)
                loss_s = criterion_s(s_out, s_lbl)
                loss = (loss_d + 0.2 * loss_s) / grad_accum

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            run_loss += loss.item() * grad_accum
            with torch.no_grad():
                preds = d_out.argmax(1)
                correct += (preds == d_lbl).sum().item()
                total += d_lbl.size(0)

        if len(train_loader) % grad_accum != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        # Validate
        val_loss, val_preds, val_targets, val_probs = evaluate(
            val_loader, model, criterion_d, criterion_s, device,
            desc=f'{desc} E{epoch+1}',
        )
        val_acc = 100.0 * (val_preds == val_targets).mean()
        mf1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)

        tag = ''
        if mf1 > best_f1 + 0.001:
            best_f1 = mf1
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
            tag = ' *'
        else:
            patience_ctr += 1

        if (epoch + 1) % 5 == 0 or epoch == 0 or tag:
            train_loss = run_loss / max(len(train_loader), 1)
            train_acc = 100.0 * correct / max(total, 1)
            print(f'      E{epoch+1:3d}/{args.epochs} | '
                  f'TrL={train_loss:.3f} TrA={train_acc:.1f}% | '
                  f'VL={val_loss:.3f} VA={val_acc:.1f}% | '
                  f'mF1={mf1:.4f}{tag}')

        if patience_ctr >= 10:
            print(f'      Early stopped at epoch {epoch+1}')
            break

    # Final eval with best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    _, final_preds, final_targets, final_probs = evaluate(
        val_loader, model, criterion_d, criterion_s, device,
        desc=f'{desc} final',
    )
    metrics = compute_full_metrics(final_preds, final_targets, final_probs)
    metrics['epochs_trained'] = epoch + 1

    del model, optimizer, scaler
    torch.cuda.empty_cache()

    return metrics


def train_dann_ablation(train_df, val_df, device, norm_mean, norm_std, args,
                        use_hard_mining=False, use_mixup=False,
                        use_cosine_warm_restarts=False,
                        use_label_smoothing=True,
                        desc='DANN'):
    """
    Train a DANN model variant for the ablation study.

    By toggling use_hard_mining, use_mixup, etc. we isolate each contribution.
    Label smoothing is always on for DANN variants (it is part of the baseline
    DANN config) unless explicitly disabled.
    """
    set_seed(args.seed)

    train_transform = make_transforms('train', norm_mean, norm_std)
    val_transform = make_transforms('val', norm_mean, norm_std)

    domain_map = DOMAIN_MAP_FULL.copy()
    num_domains = len(domain_map)

    train_ds = RetinalDANNv3Dataset(train_df, train_transform, domain_map)
    val_ds = RetinalDANNv3Dataset(val_df, val_transform, domain_map)

    sampler = make_weighted_sampler(
        train_df['disease_label'].values, n_classes=NUM_CLASSES,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    model = DANNMultiTaskViT(
        n_disease=NUM_CLASSES,
        n_severity=5,
        num_domains=num_domains,
        drop=0.3,
        backbone_name='vit_base_patch16_224',
    ).to(device)

    # Focal loss with class weights
    cw = compute_class_weight(
        'balanced',
        classes=np.arange(NUM_CLASSES),
        y=train_df['disease_label'].values,
    )
    alpha = torch.tensor(cw, dtype=torch.float32).to(device)
    alpha = alpha / alpha.sum() * NUM_CLASSES

    ls = 0.1 if use_label_smoothing else 0.0

    criterion_d = FocalLoss(alpha=alpha, gamma=2.0, label_smoothing=ls)
    criterion_s = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=ls)
    criterion_dom = nn.CrossEntropyLoss()

    # Per-sample loss for hard mining
    criterion_per_sample = nn.CrossEntropyLoss(reduction='none')

    # LLRD optimizer
    optimizer = get_optimizer_with_llrd(
        model, base_lr=args.lr, decay_factor=0.85, weight_decay=1e-4,
    )

    # Scheduler
    if use_cosine_warm_restarts:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=args.lr * 0.01,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
        )

    scaler = GradScaler()
    grad_accum = 2

    # Hard-mining tracking
    if use_hard_mining:
        from train_dann_v3 import HardExampleMiningWeightedSampler
        hard_sampler = HardExampleMiningWeightedSampler(
            labels=train_df['disease_label'].values,
            n_classes=NUM_CLASSES,
            hard_k=500,
            hard_factor=2,
        )
        # Replace the sampler in the loader
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size,
            sampler=hard_sampler,
            num_workers=args.workers, pin_memory=True,
        )

    # Mixup config
    mixup_alpha = 0.2 if use_mixup else 0.0
    mixup_prob = 0.5 if use_mixup else 0.0

    best_f1 = 0.0
    best_state = None
    patience_ctr = 0

    for epoch in range(args.epochs):
        lam_p = ganin_lambda(epoch, args.epochs, max_lambda=0.3)

        model.train()
        run_loss = 0.0
        correct = 0
        total = 0
        optimizer.zero_grad(set_to_none=True)

        epoch_indices = []
        epoch_losses = []
        epoch_preds = []
        epoch_targets = []

        for step, batch in enumerate(train_loader):
            imgs = batch[0].to(device, non_blocking=True)
            d_lbl = batch[1].to(device, non_blocking=True)
            s_lbl = batch[2].to(device, non_blocking=True)
            dom_lbl = batch[3].to(device, non_blocking=True)
            sample_idx = batch[4].to(device, non_blocking=True)

            # Optional mixup
            if use_mixup and random.random() < mixup_prob:
                lam_mix = np.random.beta(mixup_alpha, mixup_alpha)
                bs = imgs.size(0)
                index = torch.randperm(bs, device=device)
                mixed_imgs = lam_mix * imgs + (1 - lam_mix) * imgs[index]
                y_a, y_b = d_lbl, d_lbl[index]
            else:
                mixed_imgs = imgs
                y_a = d_lbl
                y_b = d_lbl
                lam_mix = 1.0

            with autocast('cuda'):
                d_out, s_out, dom_out = model(mixed_imgs, alpha=lam_p)

                if use_mixup and lam_mix < 1.0:
                    loss_d = (lam_mix * criterion_d(d_out, y_a) +
                              (1 - lam_mix) * criterion_d(d_out, y_b))
                else:
                    loss_d = criterion_d(d_out, y_a)

                loss_s = criterion_s(s_out, s_lbl)

                # Domain loss on original images
                f_orig = model.backbone(imgs)
                f_orig = model.drop(f_orig)
                dom_out_orig = model.domain_head(model.grl(f_orig, lam_p))
                loss_dom = criterion_dom(dom_out_orig, dom_lbl)

                loss = (loss_d + 0.2 * loss_s +
                        0.05 * lam_p * loss_dom) / grad_accum

            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            run_loss += loss.item() * grad_accum
            with torch.no_grad():
                preds = d_out.argmax(1)
                correct += (preds == d_lbl).sum().item()
                total += d_lbl.size(0)

                if use_hard_mining:
                    per_loss = criterion_per_sample(d_out.float(), d_lbl)
                    epoch_indices.append(sample_idx.cpu())
                    epoch_losses.append(per_loss.cpu())
                    epoch_preds.append(preds.cpu())
                    epoch_targets.append(d_lbl.cpu())

        if len(train_loader) % grad_accum != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if use_cosine_warm_restarts:
            scheduler.step(epoch)
        else:
            scheduler.step()

        # Update hard-mining sampler
        if use_hard_mining and epoch_indices:
            all_idx = torch.cat(epoch_indices)
            all_loss = torch.cat(epoch_losses)
            all_prd = torch.cat(epoch_preds)
            all_tgt = torch.cat(epoch_targets)
            hard_sampler.update_losses(all_idx, all_loss,
                                       preds=all_prd, targets=all_tgt)

        # Validate
        val_loss, val_preds, val_targets, val_probs = evaluate(
            val_loader, model, criterion_d, criterion_s, device,
            desc=f'{desc} E{epoch+1}',
        )
        val_acc = 100.0 * (val_preds == val_targets).mean()
        mf1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)

        tag = ''
        if mf1 > best_f1 + 0.001:
            best_f1 = mf1
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
            tag = ' *'
        else:
            patience_ctr += 1

        if (epoch + 1) % 5 == 0 or epoch == 0 or tag:
            train_loss = run_loss / max(len(train_loader), 1)
            train_acc = 100.0 * correct / max(total, 1)
            print(f'      E{epoch+1:3d}/{args.epochs} | '
                  f'TrL={train_loss:.3f} TrA={train_acc:.1f}% | '
                  f'VL={val_loss:.3f} VA={val_acc:.1f}% | '
                  f'mF1={mf1:.4f}{tag}')

        if patience_ctr >= 10:
            print(f'      Early stopped at epoch {epoch+1}')
            break

    # Final eval with best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    _, final_preds, final_targets, final_probs = evaluate(
        val_loader, model, criterion_d, criterion_s, device,
        desc=f'{desc} final',
    )
    metrics = compute_full_metrics(final_preds, final_targets, final_probs)
    metrics['epochs_trained'] = epoch + 1

    del model, optimizer, scaler
    torch.cuda.empty_cache()

    return metrics


def evaluate_existing_checkpoint(test_loader, device):
    """
    Evaluate the existing DANN-v3 production checkpoint on test set.
    This is ablation variant 5: full DANN-v3 pipeline.
    """
    if not os.path.exists(PRODUCTION_CKPT):
        print(f'    WARNING: Production checkpoint not found at {PRODUCTION_CKPT}')
        return None

    ckpt = torch.load(PRODUCTION_CKPT, map_location=device, weights_only=False)
    num_domains = ckpt.get('num_domains', 4)

    model = DANNMultiTaskViT(
        n_disease=NUM_CLASSES,
        n_severity=5,
        num_domains=num_domains,
        drop=0.3,
        backbone_name='vit_base_patch16_224',
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Use a simple focal loss for the evaluate() function
    alpha = torch.ones(NUM_CLASSES, device=device)
    criterion_d = FocalLoss(alpha=alpha, gamma=2.0, label_smoothing=0.0)
    criterion_s = nn.CrossEntropyLoss(ignore_index=-1)

    _, preds, targets, probs = evaluate(
        test_loader, model, criterion_d, criterion_s, device,
        desc='DANN-v3 full',
    )

    metrics = compute_full_metrics(preds, targets, probs)
    metrics['epochs_trained'] = ckpt.get('epoch', '?')
    metrics['source'] = 'loaded_checkpoint'

    del model
    torch.cuda.empty_cache()

    return metrics


def experiment_ablation(args, device):
    """
    Ablation study: train multiple model variants to isolate contributions.

    Variants:
      1. Base ViT       -- no DANN, no improvements, standard CE
      2. DANN only      -- DANNMultiTaskViT, focal loss, no extras
      3. DANN + mining  -- add hard example mining
      4. DANN + mixup   -- add mixup augmentation (no mining)
      5. DANN + all     -- full DANN-v3 pipeline (load existing checkpoint)
    """
    log_section('EXPERIMENT 4: Ablation Study')
    t0_total = time.time()

    print('\n  Loading data...')
    train_df, calib_df, test_df = load_data_splits()

    norm_mean, norm_std = load_norm_stats()

    # Build test loader for evaluating the production checkpoint
    val_transform = make_transforms('val', norm_mean, norm_std)
    test_ds = RetinalDANNv3Dataset(test_df, val_transform, DOMAIN_MAP_FULL)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    ablation_results = {}

    # Define ablation variants
    variants = [
        {
            'name': 'Base ViT (no DANN)',
            'key': 'base_vit',
            'type': 'base_vit',
            'description': 'ViT-Base/16 + CE loss, no domain adaptation',
        },
        {
            'name': 'DANN only',
            'key': 'dann_only',
            'type': 'dann',
            'use_hard_mining': False,
            'use_mixup': False,
            'use_cosine_warm_restarts': False,
            'description': 'DANNMultiTaskViT + focal loss + label smoothing',
        },
        {
            'name': 'DANN + hard mining',
            'key': 'dann_hard_mining',
            'type': 'dann',
            'use_hard_mining': True,
            'use_mixup': False,
            'use_cosine_warm_restarts': False,
            'description': 'DANN + hard example mining (top-500, 2x boost)',
        },
        {
            'name': 'DANN + mixup',
            'key': 'dann_mixup',
            'type': 'dann',
            'use_hard_mining': False,
            'use_mixup': True,
            'use_cosine_warm_restarts': False,
            'description': 'DANN + mixup augmentation (alpha=0.2, prob=0.5)',
        },
        {
            'name': 'DANN + all (DANN-v3)',
            'key': 'dann_v3_full',
            'type': 'checkpoint',
            'description': 'Full DANN-v3 pipeline (existing production model)',
        },
    ]

    for vi, variant in enumerate(variants):
        log_subsection(
            f'Ablation {vi+1}/{len(variants)}: {variant["name"]}'
        )
        print(f'    {variant["description"]}')
        t0_var = time.time()

        try:
            if variant['type'] == 'base_vit':
                metrics = train_base_vit(
                    train_df, test_df, device,
                    norm_mean, norm_std, args,
                    desc=f'Abl-{variant["key"]}',
                )

            elif variant['type'] == 'dann':
                metrics = train_dann_ablation(
                    train_df, test_df, device,
                    norm_mean, norm_std, args,
                    use_hard_mining=variant.get('use_hard_mining', False),
                    use_mixup=variant.get('use_mixup', False),
                    use_cosine_warm_restarts=variant.get(
                        'use_cosine_warm_restarts', False),
                    desc=f'Abl-{variant["key"]}',
                )

            elif variant['type'] == 'checkpoint':
                metrics = evaluate_existing_checkpoint(test_loader, device)
                if metrics is None:
                    print('    Skipping -- checkpoint not available')
                    continue

            else:
                print(f'    Unknown variant type: {variant["type"]}')
                continue

        except Exception as e:
            print(f'    ERROR in variant {variant["name"]}: {e}')
            traceback.print_exc()
            continue

        elapsed_var = time.time() - t0_var
        metrics['variant'] = variant['name']
        metrics['description'] = variant['description']
        metrics['time_seconds'] = float(elapsed_var)
        ablation_results[variant['key']] = metrics

        print_metrics_table(metrics, label=variant['name'])
        print(f'    Time: {format_duration(elapsed_var)}')

    # Summary comparison table
    log_subsection('Ablation Summary')

    print(f'\n    {"Variant":30s} | {"Acc(%)":>7s} | {"M-F1":>6s} | '
          f'{"M-AUC":>6s} | {"ECE":>6s} | {"Ep":>3s}')
    print(f'    {"-"*30}-+-{"-"*7}-+-{"-"*6}-+-{"-"*6}-+-{"-"*6}-+-{"-"*3}')

    for variant in variants:
        key = variant['key']
        if key not in ablation_results:
            continue
        m = ablation_results[key]
        ep = m.get('epochs_trained', '?')
        print(f'    {variant["name"]:30s} | {m["accuracy"]:7.2f} | '
              f'{m["macro_f1"]:.4f} | {m["macro_auc"]:.4f} | '
              f'{m["ece"]:.4f} | {str(ep):>3s}')

    # Compute deltas relative to base ViT
    if 'base_vit' in ablation_results:
        base_metrics = ablation_results['base_vit']
        print(f'\n    Improvement over Base ViT:')
        for key in ablation_results:
            if key == 'base_vit':
                continue
            m = ablation_results[key]
            d_acc = m['accuracy'] - base_metrics['accuracy']
            d_f1 = m['macro_f1'] - base_metrics['macro_f1']
            d_auc = m['macro_auc'] - base_metrics['macro_auc']
            name = next(v['name'] for v in variants if v['key'] == key)
            print(f'      {name:30s}: '
                  f'Acc {d_acc:+.2f}% | F1 {d_f1:+.4f} | AUC {d_auc:+.4f}')

    # Save results
    results_path = os.path.join(OUTPUT_DIR, 'ablation_results.json')
    with open(results_path, 'w') as f:
        json.dump(ablation_results, f, indent=2)
    print(f'\n    Results saved to {results_path}')

    # Generate ablation chart
    _plot_ablation_chart(ablation_results, variants)

    total_elapsed = time.time() - t0_total
    print(f'\n    Total ablation time: {format_duration(total_elapsed)}')

    return ablation_results


def _plot_ablation_chart(ablation_results, variants):
    """Generate ablation study bar chart."""
    if not ablation_results:
        return

    names = []
    accs = []
    f1s = []
    aucs = []

    for variant in variants:
        key = variant['key']
        if key not in ablation_results:
            continue
        m = ablation_results[key]
        names.append(variant['name'].replace('DANN + all (DANN-v3)', 'DANN-v3 (full)'))
        accs.append(m['accuracy'])
        f1s.append(m['macro_f1'])
        aucs.append(m['macro_auc'])

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width, accs, width, label='Accuracy (%)',
                   color='#2ecc71', alpha=0.85)
    bars2 = ax.bar(x, [f * 100 for f in f1s], width,
                   label='Macro F1 (x100)', color='#3498db', alpha=0.85)
    bars3 = ax.bar(x + width, [a * 100 for a in aucs], width,
                   label='Macro AUC (x100)', color='#e74c3c', alpha=0.85)

    ax.set_xlabel('Model Variant', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Ablation Study: Component Contributions',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    # Value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, 'ablation_chart.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Ablation chart saved to {chart_path}')


# ================================================================
# EXPERIMENT 5: CONFIDENCE ROUTING EVALUATION
# ================================================================
def experiment_eval_routing():
    """Run confidence routing evaluation."""
    log_section('EXPERIMENT 5: Confidence Routing Evaluation')
    t0 = time.time()

    success = run_subprocess('confidence_routing.py', 'Confidence routing')
    elapsed = time.time() - t0

    if success:
        print(f'\n    Confidence routing eval completed in {format_duration(elapsed)}')
    return success


# ================================================================
# MAIN
# ================================================================
def main():
    args = parse_args()

    # Determine which experiments to run
    run_faiss = args.rebuild_faiss or args.all
    run_rad = args.eval_rad or args.all
    run_lodo = args.lodo or args.all
    run_ablation = args.ablation or args.all
    run_routing = args.eval_routing or args.all

    if not any([run_faiss, run_rad, run_lodo, run_ablation, run_routing]):
        print('No experiments selected. Use --all or specific flags.')
        print('Run with --help for usage information.')
        return

    # Setup
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log_section('RetinaSense IEEE Paper Experiments', char='#', width=75)
    print(f'  Date       : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'  Device     : {device}')
    if torch.cuda.is_available():
        print(f'  GPU        : {torch.cuda.get_device_name(args.gpu)}')
        vram = torch.cuda.get_device_properties(args.gpu).total_memory / 1e9
        print(f'  VRAM       : {vram:.1f} GB')
    print(f'  Seed       : {args.seed}')
    print(f'  Epochs     : {args.epochs} (for LODO/ablation)')
    print(f'  Batch size : {args.batch_size}')
    print(f'  LR         : {args.lr:.1e}')
    print()

    experiments_to_run = []
    if run_faiss:
        experiments_to_run.append('1-FAISS Rebuild')
    if run_rad:
        experiments_to_run.append('2-RAD Evaluation')
    if run_lodo:
        experiments_to_run.append('3-LODO Validation')
    if run_ablation:
        experiments_to_run.append('4-Ablation Study')
    if run_routing:
        experiments_to_run.append('5-Confidence Routing')
    print(f'  Experiments: {", ".join(experiments_to_run)}')

    t_total_start = time.time()
    results_summary = {}

    # Experiment 1: Rebuild FAISS
    if run_faiss:
        try:
            success = experiment_rebuild_faiss()
            results_summary['faiss'] = {'status': 'ok' if success else 'failed'}
        except Exception as e:
            print(f'  ERROR in FAISS rebuild: {e}')
            traceback.print_exc()
            results_summary['faiss'] = {'status': 'error', 'message': str(e)}

    # Experiment 2: RAD Evaluation
    if run_rad:
        try:
            success = experiment_eval_rad()
            results_summary['rad'] = {'status': 'ok' if success else 'failed'}
        except Exception as e:
            print(f'  ERROR in RAD evaluation: {e}')
            traceback.print_exc()
            results_summary['rad'] = {'status': 'error', 'message': str(e)}

    # Experiment 3: LODO
    if run_lodo:
        try:
            lodo_results = experiment_lodo(args, device)
            results_summary['lodo'] = {'status': 'ok', 'results_file': 'outputs_v3/lodo_results.json'}
        except Exception as e:
            print(f'  ERROR in LODO: {e}')
            traceback.print_exc()
            results_summary['lodo'] = {'status': 'error', 'message': str(e)}

    # Experiment 4: Ablation
    if run_ablation:
        try:
            ablation_results = experiment_ablation(args, device)
            results_summary['ablation'] = {'status': 'ok', 'results_file': 'outputs_v3/ablation_results.json'}
        except Exception as e:
            print(f'  ERROR in ablation: {e}')
            traceback.print_exc()
            results_summary['ablation'] = {'status': 'error', 'message': str(e)}

    # Experiment 5: Confidence Routing
    if run_routing:
        try:
            success = experiment_eval_routing()
            results_summary['routing'] = {'status': 'ok' if success else 'failed'}
        except Exception as e:
            print(f'  ERROR in routing eval: {e}')
            traceback.print_exc()
            results_summary['routing'] = {'status': 'error', 'message': str(e)}

    # Final summary
    total_elapsed = time.time() - t_total_start
    log_section('EXPERIMENT SUITE COMPLETE', char='#', width=75)

    print(f'\n  Total time: {format_duration(total_elapsed)}')
    print(f'\n  Results:')
    for exp_name, result in results_summary.items():
        status = result['status']
        extra = f' -> {result["results_file"]}' if 'results_file' in result else ''
        print(f'    {exp_name:12s}: {status}{extra}')

    # Save master summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_time_seconds': float(total_elapsed),
        'args': vars(args),
        'experiments': results_summary,
    }
    summary_path = os.path.join(OUTPUT_DIR, 'paper_experiments_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n  Master summary saved to {summary_path}')

    print(f'\n  Output files:')
    for fname in ['lodo_results.json', 'lodo_chart.png',
                   'ablation_results.json', 'ablation_chart.png',
                   'paper_experiments_summary.json']:
        fpath = os.path.join(OUTPUT_DIR, fname)
        exists = os.path.exists(fpath)
        marker = '[OK]' if exists else '[--]'
        print(f'    {marker} {fpath}')

    print()


if __name__ == '__main__':
    main()
