#!/usr/bin/env python3
"""
RetinaSense v3.0 — Phase 2A: 5-Fold Stratified Cross-Validation
================================================================
Trains 5 independent models on different stratified splits.
Reports mean ± std for all metrics → confidence intervals for paper.

Run time estimate: ~2 hrs on H100 (25 epochs x 5 folds)
"""

import os, json, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (f1_score, classification_report,
                              roc_auc_score, accuracy_score,
                              balanced_accuracy_score)
from sklearn.preprocessing import label_binarize
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ================================================================
# CONFIG
# ================================================================
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE   = 224
N_CLASSES  = 5
CLASS_NAMES= ['Normal', 'Diabetes/DR', 'Glaucoma', 'Cataract', 'AMD']
N_FOLDS    = 5
N_EPOCHS   = 30       # enough to converge; early stopping at patience=8
PATIENCE   = 8
BATCH_SIZE = 32
BASE_LR    = 3e-4
LLRD_DECAY = 0.75
WEIGHT_DECAY = 1e-4
MIXUP_ALPHA  = 0.4
FOCAL_GAMMA  = 1.0
GRAD_ACCUM   = 2
NUM_WORKERS  = 4
CACHE_DIR  = './preprocessed_cache_unified'
OUTPUT_DIR = './outputs_v3/kfold'
os.makedirs(OUTPUT_DIR, exist_ok=True)

_norm_candidates = ['configs/fundus_norm_stats_unified.json', 'configs/fundus_norm_stats.json', 'data/fundus_norm_stats.json']
for _np in _norm_candidates:
    if os.path.exists(_np):
        with open(_np) as f:
            ns = json.load(f)
        print(f'  Norm stats from {_np}')
        break
NORM_MEAN, NORM_STD = ns['mean_rgb'], ns['std_rgb']

print('=' * 65)
print('  RetinaSense v3.0 — 5-Fold Cross-Validation')
print('=' * 65)
print(f'  Device : {DEVICE}')
if torch.cuda.is_available():
    print(f'  GPU    : {torch.cuda.get_device_name(0)}')

# ================================================================
# FULL DATASET (train + calib merged for CV)
# ================================================================
train_df = pd.read_csv('./data/train_split.csv')
calib_df = pd.read_csv('./data/calib_split.csv')
full_df  = pd.concat([train_df, calib_df], ignore_index=True)
test_df  = pd.read_csv('./data/test_split.csv')  # sealed — only for final eval

print(f'\n  Full CV pool : {len(full_df)} samples')
print(f'  Test set     : {len(test_df)} samples (sealed — used only at end)')
for c, cn in enumerate(CLASS_NAMES):
    n = (full_df['disease_label'] == c).sum()
    print(f'    {cn:15s}: {n} ({100*n/len(full_df):.1f}%)')


# ================================================================
# DATASET
# ================================================================
def _cache_key(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(CACHE_DIR, f'{stem}_{IMG_SIZE}.npy')

def make_transforms(phase):
    norm = transforms.Normalize(NORM_MEAN, NORM_STD)
    if phase == 'train':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(), norm,
            transforms.RandomErasing(p=0.2),
        ])
    return transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), norm])

class RetinalDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cache_fp = row.get('cache_path', _cache_key(row['image_path']))
        # Try unified cache first
        unified_fp = cache_fp.replace('preprocessed_cache_v3', 'preprocessed_cache_unified')
        if os.path.exists(unified_fp):
            cache_fp = unified_fp
        try:
            img = np.load(cache_fp)
        except Exception:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        sev = int(row.get('severity_label', -1))
        if sev < 0: sev = 0
        return (self.transform(img),
                torch.tensor(int(row['disease_label']), dtype=torch.long),
                torch.tensor(sev, dtype=torch.long))

def make_weighted_sampler(df):
    labels = df['disease_label'].values
    cnts   = np.bincount(labels, minlength=N_CLASSES).astype(float)
    cnts   = np.where(cnts == 0, 1.0, cnts)
    w      = 1.0 / cnts[labels]
    return WeightedRandomSampler(torch.DoubleTensor(w), len(w), replacement=True)


# ================================================================
# MODEL (same MultiTaskViT as training)
# ================================================================
class MultiTaskViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        self.drop = nn.Dropout(0.3)
        self.disease_head = nn.Sequential(
            nn.Linear(768, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, N_CLASSES))
        self.severity_head = nn.Sequential(
            nn.Linear(768, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 5))
    def forward(self, x):
        f = self.backbone(x); f = self.drop(f)
        return self.disease_head(f), self.severity_head(f)

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        if alpha is not None: self.register_buffer('alpha', alpha)
        else: self.alpha = None
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            focal = self.alpha.gather(0, targets) * focal
        return focal.mean()

def get_llrd_optimizer(model):
    groups = []
    groups.append({'params': list(model.disease_head.parameters()) +
                              list(model.severity_head.parameters()) +
                              list(model.drop.parameters()), 'lr': BASE_LR})
    for i in range(len(model.backbone.blocks) - 1, -1, -1):
        dist = len(model.backbone.blocks) - i
        groups.append({'params': list(model.backbone.blocks[i].parameters()),
                       'lr': BASE_LR * (LLRD_DECAY ** dist)})
    embed_lr = BASE_LR * (LLRD_DECAY ** (len(model.backbone.blocks) + 1))
    groups.append({'params': list(model.backbone.patch_embed.parameters()) +
                              [model.backbone.cls_token, model.backbone.pos_embed] +
                              list(model.backbone.norm.parameters()), 'lr': embed_lr})
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)

def mixup(x, y, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def _ece(probs, labels, n_bins=15):
    confs = probs.max(axis=1); preds = probs.argmax(axis=1); acc = preds == labels
    edges = np.linspace(0, 1, n_bins + 1); ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confs >= lo) & (confs < hi)
        if m.sum() > 0:
            ece += m.sum() * abs(acc[m].mean() - confs[m].mean())
    return float(ece / len(labels))

@torch.no_grad()
def evaluate(loader, model, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for imgs, d_lbl, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        with autocast('cuda'):
            d_out, _ = model(imgs)
        probs = torch.softmax(d_out.float(), dim=1)
        all_preds.extend(d_out.argmax(1).cpu().numpy())
        all_labels.extend(d_lbl.numpy())
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


# ================================================================
# CROSS-VALIDATION LOOP
# ================================================================
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
labels_for_split = full_df['disease_label'].values

fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(full_df, labels_for_split)):
    print(f'\n{"="*65}')
    print(f'  FOLD {fold+1}/{N_FOLDS}')
    print(f'{"="*65}')

    fold_train = full_df.iloc[train_idx]
    fold_val   = full_df.iloc[val_idx]
    print(f'  Train: {len(fold_train)} | Val: {len(fold_val)}')

    # Datasets
    train_ds = RetinalDataset(fold_train, make_transforms('train'))
    val_ds   = RetinalDataset(fold_val,   make_transforms('val'))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=make_weighted_sampler(fold_train),
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True)

    # Model + optimizer
    model = MultiTaskViT().to(DEVICE)
    cw    = compute_class_weight('balanced',
                                  classes=np.arange(N_CLASSES),
                                  y=fold_train['disease_label'].values)
    alpha = torch.tensor(cw / cw.sum() * N_CLASSES, dtype=torch.float32).to(DEVICE)
    crit_d = FocalLoss(alpha=alpha, gamma=FOCAL_GAMMA)
    crit_s = nn.CrossEntropyLoss(ignore_index=-1)
    opt    = get_llrd_optimizer(model)
    sched  = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g['lr'] for g in opt.param_groups],
        steps_per_epoch=len(train_loader), epochs=N_EPOCHS,
        pct_start=0.1, anneal_strategy='cos',
        div_factor=10.0, final_div_factor=100.0)
    scaler = GradScaler()

    best_f1, best_state, patience_ctr = 0.0, None, 0
    t_fold = time.time()

    for epoch in range(N_EPOCHS):
        model.train()
        run_loss = correct = total = 0
        opt.zero_grad(set_to_none=True)

        for step, (imgs, d_lbl, s_lbl) in enumerate(train_loader):
            imgs  = imgs.to(DEVICE, non_blocking=True)
            d_lbl = d_lbl.to(DEVICE, non_blocking=True)
            s_lbl = s_lbl.to(DEVICE, non_blocking=True)

            mx, ya, yb, lam = mixup(imgs, d_lbl)
            with autocast('cuda'):
                d_out, s_out = model(mx)
                loss = (lam * crit_d(d_out, ya) + (1-lam) * crit_d(d_out, yb)
                        + 0.2 * crit_s(s_out, s_lbl)) / GRAD_ACCUM

            if not (torch.isnan(loss) or torch.isinf(loss)):
                scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step()
                opt.zero_grad(set_to_none=True)

            run_loss += loss.item() * GRAD_ACCUM
            with torch.no_grad():
                correct += (d_out.argmax(1) == ya).sum().item()
                total   += d_lbl.size(0)

        # Flush
        if len(train_loader) % GRAD_ACCUM != 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            opt.zero_grad(set_to_none=True)

        preds, targets, probs = evaluate(val_loader, model, DEVICE)
        mf1 = f1_score(targets, preds, average='macro')
        acc = 100 * (preds == targets).mean()
        per = f1_score(targets, preds, average=None, labels=range(N_CLASSES), zero_division=0)

        tag = ''
        if mf1 > best_f1 + 0.001:
            best_f1 = mf1; patience_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            tag = f' * BEST'
        else:
            patience_ctr += 1

        cls = ' | '.join(f'{cn[:3]}:{per[ci]:.2f}' for ci, cn in enumerate(CLASS_NAMES))
        print(f'  F{fold+1} E{epoch+1:02d} | Loss {run_loss/len(train_loader):.3f} | '
              f'Acc {acc:.1f}% | mF1 {mf1:.4f}{tag}')
        print(f'     {cls}')

        if patience_ctr >= PATIENCE:
            print(f'  Early stop at epoch {epoch+1}')
            break

    fold_time = (time.time() - t_fold) / 60
    print(f'  Fold {fold+1} done in {fold_time:.1f} min. Best val F1: {best_f1:.4f}')

    # Save fold checkpoint
    ck_path = os.path.join(OUTPUT_DIR, f'fold_{fold+1}_best.pth')
    torch.save({'fold': fold+1, 'macro_f1': best_f1, 'model_state_dict': best_state}, ck_path)

    # Final eval with best model on fold val set
    model.load_state_dict(best_state)
    preds, targets, probs = evaluate(val_loader, model, DEVICE)

    # Temperature calibration on val set
    all_logits, all_tgts = [], []
    model.eval()
    with torch.no_grad():
        for imgs, d_lbl, _ in val_loader:
            imgs = imgs.to(DEVICE)
            d_out, _ = model(imgs)
            all_logits.append(d_out.float().cpu())
            all_tgts.append(d_lbl)
    logits_t = torch.cat(all_logits)
    labels_t = torch.cat(all_tgts)
    res = minimize_scalar(
        lambda T: F.nll_loss(F.log_softmax(logits_t / T, dim=1), labels_t).item(),
        bounds=(0.01, 10.0), method='bounded')
    T_fold = float(res.x)
    cal_probs = torch.softmax(logits_t / T_fold, dim=1).numpy()
    cal_preds = cal_probs.argmax(axis=1)

    acc   = accuracy_score(targets, cal_preds)
    bal   = balanced_accuracy_score(targets, cal_preds)
    mf1   = f1_score(targets, cal_preds, average='macro')
    wf1   = f1_score(targets, cal_preds, average='weighted')
    per   = f1_score(targets, cal_preds, average=None, labels=range(N_CLASSES), zero_division=0)
    ece   = _ece(cal_probs, targets)
    y_bin = label_binarize(targets, classes=list(range(N_CLASSES)))
    try:
        mauc = roc_auc_score(y_bin, cal_probs, multi_class='ovr', average='macro')
    except:
        mauc = 0.0

    m = {'fold': fold+1, 'temperature': T_fold,
         'accuracy': float(acc), 'balanced_accuracy': float(bal),
         'macro_f1': float(mf1), 'weighted_f1': float(wf1),
         'macro_auc': float(mauc), 'ece': float(ece),
         **{f'f1_{CLASS_NAMES[i]}': float(per[i]) for i in range(N_CLASSES)}}
    fold_metrics.append(m)
    print(f'\n  Fold {fold+1} Final: Acc={acc*100:.1f}% | Bal={bal*100:.1f}% | mF1={mf1:.4f} | AUC={mauc:.4f}')


# ================================================================
# AGGREGATE RESULTS
# ================================================================
print('\n' + '=' * 65)
print('  CROSS-VALIDATION RESULTS')
print('=' * 65)

keys = ['accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1', 'macro_auc', 'ece',
        *[f'f1_{cn}' for cn in CLASS_NAMES]]

agg = {}
for k in keys:
    vals = [m[k] for m in fold_metrics]
    agg[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
               'min': float(np.min(vals)), 'max': float(np.max(vals)),
               'values': vals}
    print(f'  {k:25s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}  '
          f'[{np.min(vals):.4f}, {np.max(vals):.4f}]')

# Save results
results = {'n_folds': N_FOLDS, 'fold_metrics': fold_metrics, 'aggregate': agg}
with open(os.path.join(OUTPUT_DIR, 'kfold_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f'\n  Results saved -> {OUTPUT_DIR}/kfold_results.json')


# ================================================================
# PLOT: fold comparison
# ================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = plt.cm.tab10(np.linspace(0, 1, N_FOLDS))

# Per-fold bar chart
metrics_to_plot = ['accuracy', 'macro_f1', 'macro_auc']
labels_plot     = ['Accuracy', 'Macro F1', 'Macro AUC']
folds = [f'Fold {i+1}' for i in range(N_FOLDS)]

for ax, mk, ml in zip(axes, metrics_to_plot, labels_plot):
    vals = [m[mk] for m in fold_metrics]
    bars = ax.bar(folds, vals, color=colors, alpha=0.8, edgecolor='black')
    mean_v = np.mean(vals)
    ax.axhline(mean_v, color='red', linestyle='--', lw=2, label=f'Mean={mean_v:.3f}')
    ax.fill_between(range(N_FOLDS),
                    [mean_v - np.std(vals)] * N_FOLDS,
                    [mean_v + np.std(vals)] * N_FOLDS,
                    alpha=0.15, color='red', label=f'±1σ={np.std(vals):.3f}')
    ax.set_title(f'{ml} per Fold', fontweight='bold')
    ax.set_ylim(max(0, min(vals) - 0.05), min(1, max(vals) + 0.05))
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=8)

plt.suptitle(f'RetinaSense-ViT — {N_FOLDS}-Fold CV Summary\n'
             f'Accuracy: {agg["accuracy"]["mean"]:.3f}±{agg["accuracy"]["std"]:.3f} | '
             f'Macro F1: {agg["macro_f1"]["mean"]:.3f}±{agg["macro_f1"]["std"]:.3f} | '
             f'AUC: {agg["macro_auc"]["mean"]:.3f}±{agg["macro_auc"]["std"]:.3f}',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fold_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# Per-class F1 boxplot across folds
fig, ax = plt.subplots(figsize=(10, 5))
class_f1s = [[m[f'f1_{cn}'] for m in fold_metrics] for cn in CLASS_NAMES]
bp = ax.boxplot(class_f1s, labels=CLASS_NAMES, patch_artist=True)
for patch, color in zip(bp['boxes'], plt.cm.Set2(np.linspace(0, 1, N_CLASSES))):
    patch.set_facecolor(color)
ax.set_title(f'Per-Class F1 Distribution Across {N_FOLDS} Folds', fontweight='bold')
ax.set_ylabel('F1 Score'); ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'perclass_f1_boxplot.png'), dpi=150)
plt.close()

print(f'\n  Plots saved -> {OUTPUT_DIR}/')
print('=' * 65)
print('  5-FOLD CV COMPLETE')
print(f'  Accuracy : {agg["accuracy"]["mean"]*100:.2f}% ± {agg["accuracy"]["std"]*100:.2f}%')
print(f'  Macro F1 : {agg["macro_f1"]["mean"]:.4f} ± {agg["macro_f1"]["std"]:.4f}')
print(f'  Macro AUC: {agg["macro_auc"]["mean"]:.4f} ± {agg["macro_auc"]["std"]:.4f}')
print('=' * 65)
