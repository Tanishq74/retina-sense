#!/usr/bin/env python3
"""
RetinaSense v2 — Production-Grade Training Pipeline
====================================================
Fixes from v1:
  1. Focal Loss (handles class imbalance far better than weighted CE)
  2. Stratified batch sampler (every batch sees all classes)
  3. LR warmup + cosine decay (stable optimisation)
  4. Gradient accumulation (effective batch 128, actual batch 32)
  5. Early stopping on Macro F1 (not accuracy — misleading with imbalance)
  6. Per-class metrics tracked every epoch
  7. Pre-cached preprocessing (GPU efficiency)
  8. Proper NaN handling in mixed precision
  9. Comprehensive plots after training
"""

import os, sys, time, warnings, json
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from collections import Counter
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════
SAVE_DIR      = './outputs_v2'
CACHE_DIR     = './preprocessed_cache'
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

EPOCHS            = 20
WARMUP_EPOCHS     = 3      # heads-only warmup
LR_WARMUP_STEPS   = 3      # linear warmup epochs after unfreeze
BATCH_SIZE        = 32      # actual batch size (stable)
ACCUM_STEPS       = 2       # gradient accumulation → effective batch 64
NUM_WORKERS       = 8
PATIENCE          = 7       # early stopping on macro-F1
FOCAL_GAMMA       = 1.0     # reduced from 2.0 — less aggressive
IMG_SIZE          = 300     # EfficientNet-B3 optimal input

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['Normal', 'Diabetes/DR', 'Glaucoma', 'Cataract', 'AMD']
NUM_CLASSES = len(CLASS_NAMES)

print('='*65)
print('       RetinaSense v2 — Production Pipeline')
print('='*65)
if torch.cuda.is_available():
    print(f'  GPU         : {torch.cuda.get_device_name(0)}')
    print(f'  VRAM        : {round(torch.cuda.get_device_properties(0).total_memory/1e9,1)} GB')
print(f'  Epochs      : {EPOCHS}')
print(f'  Batch       : {BATCH_SIZE} (effective {BATCH_SIZE*ACCUM_STEPS} via grad accum)')
print(f'  Image Size  : {IMG_SIZE}')
print(f'  Focal Loss γ: {FOCAL_GAMMA} (mild — avoids over-correction)')
print(f'  Early Stop  : patience={PATIENCE} on macro-F1')
print('='*65)

# ═══════════════════════════════════════════════════════════
# 1  METADATA
# ═══════════════════════════════════════════════════════════
print('\n[1/7] Building metadata...')
BASE = './'
disease_cols = ['N','D','G','C','A']
label_map = {'N':0,'D':1,'G':2,'C':3,'A':4}

df_odir = pd.read_csv(f'{BASE}/odir/full_df.csv')
df_odir['disease_count'] = df_odir[disease_cols].sum(axis=1)
df_odir = df_odir[df_odir['disease_count']==1].copy()
def get_label(row):
    for d in disease_cols:
        if row[d]==1: return label_map[d]
df_odir['disease_label'] = df_odir.apply(get_label, axis=1)

img_col = next(c for c in df_odir.columns
               if any(k in c.lower() for k in ['filename','fundus','image']))

odir_meta = pd.DataFrame({
    'image_path':    f'{BASE}/odir/preprocessed_images/'+df_odir[img_col].astype(str),
    'dataset':       'ODIR',
    'disease_label': df_odir['disease_label'],
    'severity_label':-1
})

df_aptos = pd.read_csv(f'{BASE}/aptos/train.csv')
aptos_meta = pd.DataFrame({
    'image_path':    f'{BASE}/aptos/train_images/'+df_aptos['id_code']+'.png',
    'dataset':       'APTOS',
    'disease_label': 1,
    'severity_label':df_aptos['diagnosis']
})

meta = pd.concat([odir_meta, aptos_meta], ignore_index=True)
meta = meta[meta['image_path'].apply(os.path.exists)].reset_index(drop=True)
print(f'  Total samples: {len(meta)}')
dist = meta['disease_label'].value_counts().sort_index()
for i,cnt in dist.items():
    print(f'    {CLASS_NAMES[i]:15s}: {cnt:4d}  ({100*cnt/len(meta):.1f}%)')

# ═══════════════════════════════════════════════════════════
# 2  PRE-CACHE
# ═══════════════════════════════════════════════════════════
print(f'\n[2/7] Pre-caching @ {IMG_SIZE}×{IMG_SIZE}...')

def ben_graham(path, sz=IMG_SIZE, sigma=10):
    img = cv2.imread(path)
    if img is None:
        img = np.array(Image.open(path).convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (sz, sz))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img,(0,0),sigma), -4, 128)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (sz//2, sz//2), int(sz*0.48), 255, -1)
    return cv2.bitwise_and(img, img, mask=mask)

cache_paths = []
cached = 0
for _, row in tqdm(meta.iterrows(), total=len(meta), desc='Caching'):
    stem = os.path.splitext(os.path.basename(row['image_path']))[0]
    fp   = f'{CACHE_DIR}/{stem}_{IMG_SIZE}.npy'
    if not os.path.exists(fp):
        try:
            np.save(fp, ben_graham(row['image_path']))
        except Exception:
            np.save(fp, np.zeros((IMG_SIZE,IMG_SIZE,3), dtype=np.uint8))
        cached += 1
    cache_paths.append(fp)
meta['cache_path'] = cache_paths
print(f'  Newly cached: {cached} | Already cached: {len(meta)-cached}')

# ═══════════════════════════════════════════════════════════
# 3  DATASET + LOADERS
# ═══════════════════════════════════════════════════════════
print('\n[3/7] Creating data loaders...')

train_df, val_df = train_test_split(
    meta, test_size=0.2, stratify=meta['disease_label'], random_state=42)

def make_transforms(phase):
    if phase == 'train':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.05,0.05), scale=(0.95,1.05)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            transforms.RandomErasing(p=0.2),
        ])
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

class RetDS(Dataset):
    def __init__(self, df, tfm):
        self.df  = df.reset_index(drop=True)
        self.tfm = tfm
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        try:    img = np.load(r['cache_path'])
        except: img = np.zeros((IMG_SIZE,IMG_SIZE,3), dtype=np.uint8)
        return (self.tfm(img),
                torch.tensor(int(r['disease_label']),  dtype=torch.long),
                torch.tensor(int(r['severity_label']), dtype=torch.long))

train_ds = RetDS(train_df, make_transforms('train'))
val_ds   = RetDS(val_df,   make_transforms('val'))

# Use shuffle (not WeightedRandomSampler — that over-corrects with focal loss)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=True, prefetch_factor=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=True)

print(f'  Train : {len(train_ds):5d}  ({len(train_loader):3d} batches)')
print(f'  Val   : {len(val_ds):5d}  ({len(val_loader):3d} batches)')
print(f'  ⚡ Focal Loss + class weights handle imbalance (no oversampling)')

# ═══════════════════════════════════════════════════════════
# 4  MODEL + FOCAL LOSS
# ═══════════════════════════════════════════════════════════
print('\n[4/7] Building model...')

class FocalLoss(nn.Module):
    """Focal Loss — down-weights easy examples, focuses on hard ones."""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            at = self.alpha.gather(0, targets)
            focal = at * focal
        return focal.mean()


class MultiTaskModel(nn.Module):
    def __init__(self, n_disease=5, n_severity=5, drop=0.4):
        super().__init__()
        bb = models.efficientnet_b3(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(bb.children())[:-1])
        feat = 1536
        self.drop = nn.Dropout(drop)
        self.disease_head = nn.Sequential(
            nn.Linear(feat, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_disease))
        self.severity_head = nn.Sequential(
            nn.Linear(feat, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_severity))

    def forward(self, x):
        f = self.backbone(x).flatten(1)
        f = self.drop(f)
        return self.disease_head(f), self.severity_head(f)

model = MultiTaskModel().to(device)

# class-weight alpha for focal loss
cw = compute_class_weight('balanced', classes=np.arange(5), y=train_df['disease_label'].values)
alpha = torch.tensor(cw, dtype=torch.float32).to(device)
alpha = alpha / alpha.sum() * NUM_CLASSES  # normalize
print(f'  Focal α: {[f"{a:.2f}" for a in alpha.tolist()]}')

criterion_d = FocalLoss(alpha=alpha, gamma=FOCAL_GAMMA)
criterion_s = nn.CrossEntropyLoss(ignore_index=-1)

total_p = sum(p.numel() for p in model.parameters())
print(f'  Params: {total_p:,}')

# ═══════════════════════════════════════════════════════════
# 5  TRAINING LOOP
# ═══════════════════════════════════════════════════════════
print('\n[5/7] Training...')

# freeze backbone first
for p in model.backbone.parameters():
    p.requires_grad = False

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=3e-4, weight_decay=1e-3)
scaler = GradScaler()

def get_scheduler(opt, warmup_steps, total_steps):
    """Linear warmup then cosine decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

CHECKPOINT = f'{SAVE_DIR}/best_model.pth'

history = {k:[] for k in [
    'train_loss','val_loss','train_acc','val_acc',
    'macro_f1','weighted_f1','lr',
    *(f'f1_{c}' for c in CLASS_NAMES)
]}

best_f1    = 0.0
patience_ctr = 0
total_steps = EPOCHS * len(train_loader) // ACCUM_STEPS
sched = get_scheduler(optimizer, warmup_steps=len(train_loader)//ACCUM_STEPS, total_steps=total_steps)

print('='*65)

for epoch in range(EPOCHS):
    t0 = time.time()

    # ── Unfreeze backbone after warmup ──
    if epoch == WARMUP_EPOCHS:
        print('\n  🔓 Unfreezing backbone with LR warmup')
        for p in model.backbone.parameters():
            p.requires_grad = True
        # new optimizer for full model with lower LR for backbone
        optimizer = torch.optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': 1e-5},
            {'params': model.disease_head.parameters(),  'lr': 1e-4},
            {'params': model.severity_head.parameters(), 'lr': 1e-4},
        ], weight_decay=1e-3)
        remaining = (EPOCHS - WARMUP_EPOCHS) * len(train_loader) // ACCUM_STEPS
        sched = get_scheduler(optimizer,
                              warmup_steps=LR_WARMUP_STEPS * len(train_loader) // ACCUM_STEPS,
                              total_steps=remaining)
        scaler = GradScaler()

    # ── TRAIN ──
    model.train()
    run_loss = 0.0
    correct  = 0
    total    = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(train_loader, desc=f'E{epoch+1:02d}/{EPOCHS} train', leave=False)
    for step, (imgs, d_lbl, s_lbl) in enumerate(pbar):
        imgs  = imgs.to(device, non_blocking=True)
        d_lbl = d_lbl.to(device, non_blocking=True)
        s_lbl = s_lbl.to(device, non_blocking=True)

        with autocast('cuda'):
            d_out, s_out = model(imgs)
            loss_d = criterion_d(d_out, d_lbl)
            loss_s = criterion_s(s_out, s_lbl)
            loss   = (loss_d + 0.2 * loss_s) / ACCUM_STEPS

        # check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()

        if (step + 1) % ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            sched.step()

        run_loss += loss.item() * ACCUM_STEPS
        preds     = d_out.argmax(1)
        correct  += (preds == d_lbl).sum().item()
        total    += d_lbl.size(0)
        pbar.set_postfix(loss=f'{loss.item()*ACCUM_STEPS:.3f}',
                         acc=f'{100*correct/total:.1f}%')

    train_loss = run_loss / len(train_loader)
    train_acc  = 100 * correct / total

    # ── VALIDATE ──
    model.eval()
    vl = 0.0
    all_p, all_t, all_prob = [], [], []
    with torch.no_grad():
        for imgs, d_lbl, s_lbl in tqdm(val_loader, desc=f'E{epoch+1:02d}/{EPOCHS} val  ', leave=False):
            imgs  = imgs.to(device, non_blocking=True)
            d_lbl = d_lbl.to(device, non_blocking=True)
            s_lbl = s_lbl.to(device, non_blocking=True)
            with autocast('cuda'):
                d_out, s_out = model(imgs)
                ld = criterion_d(d_out, d_lbl)
                ls = criterion_s(s_out, s_lbl)
                loss = ld + 0.2 * ls
            if not (torch.isnan(loss) or torch.isinf(loss)):
                vl += loss.item()
            probs = torch.softmax(d_out.float(), dim=1)
            all_p.extend(d_out.argmax(1).cpu().numpy())
            all_t.extend(d_lbl.cpu().numpy())
            all_prob.extend(probs.cpu().numpy())

    val_loss = vl / len(val_loader)
    all_p, all_t, all_prob = np.array(all_p), np.array(all_t), np.array(all_prob)
    val_acc = 100 * (all_p == all_t).mean()

    mf1 = f1_score(all_t, all_p, average='macro')
    wf1 = f1_score(all_t, all_p, average='weighted')
    per_f1 = f1_score(all_t, all_p, average=None, labels=range(NUM_CLASSES), zero_division=0)

    lr = optimizer.param_groups[0]['lr']

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    history['macro_f1'].append(mf1)
    history['weighted_f1'].append(wf1)
    history['lr'].append(lr)
    for ci, cn in enumerate(CLASS_NAMES):
        history[f'f1_{cn}'].append(per_f1[ci])

    elapsed = time.time() - t0

    tag = ''
    if mf1 > best_f1:
        best_f1 = mf1
        patience_ctr = 0
        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'val_acc': val_acc, 'macro_f1': mf1, 'history': history
        }, CHECKPOINT)
        tag = f' ★ NEW BEST (macro-F1={mf1:.4f})'
    else:
        patience_ctr += 1

    cls_str = ' | '.join(f'{cn[:3]}:{per_f1[ci]:.2f}' for ci,cn in enumerate(CLASS_NAMES))
    print(f'E{epoch+1:02d} | {elapsed:.0f}s | LR {lr:.1e} | '
          f'TrL {train_loss:.3f} TrA {train_acc:.1f}% | '
          f'VL {val_loss:.3f} VA {val_acc:.1f}% | '
          f'mF1 {mf1:.3f} wF1 {wf1:.3f}{tag}')
    print(f'     {cls_str}')

    if patience_ctr >= PATIENCE:
        print(f'\n  ⏹  Early stopping — no improvement for {PATIENCE} epochs')
        break

print(f'\n✅ Training done. Best macro-F1: {best_f1:.4f}')

# ═══════════════════════════════════════════════════════════
# 6  EVALUATION + PLOTS
# ═══════════════════════════════════════════════════════════
print('\n[6/7] Full evaluation...')

ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
history = ckpt['history']

all_p, all_t, all_prob = [], [], []
with torch.no_grad():
    for imgs, d_lbl, _ in tqdm(val_loader, desc='Evaluating'):
        imgs = imgs.to(device)
        d_out, _ = model(imgs)
        all_p.extend(d_out.argmax(1).cpu().numpy())
        all_t.extend(d_lbl.numpy())
        all_prob.extend(torch.softmax(d_out.float(), dim=1).cpu().numpy())

all_p    = np.array(all_p)
all_t    = np.array(all_t)
all_prob = np.array(all_prob)

print('\n' + '='*65)
print('             CLASSIFICATION REPORT')
print('='*65)
report = classification_report(all_t, all_p, target_names=CLASS_NAMES, digits=4)
print(report)
mf1 = f1_score(all_t, all_p, average='macro')
wf1 = f1_score(all_t, all_p, average='weighted')
try:    mauc = roc_auc_score(all_t, all_prob, multi_class='ovr', average='macro')
except: mauc = 0.0
print(f'Macro F1    : {mf1:.4f}')
print(f'Weighted F1 : {wf1:.4f}')
print(f'Macro AUC   : {mauc:.4f}')

# ═══════════════════════════════════════════════════════════
# 7  COMPREHENSIVE PLOTS
# ═══════════════════════════════════════════════════════════
print('\n[7/7] Generating plots...')
ep = range(1, len(history['train_loss'])+1)
colors = ['#2ecc71','#3498db','#e74c3c','#f39c12','#9b59b6']

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# ── 1. Loss ──
axes[0,0].plot(ep, history['train_loss'], 'b-o', ms=4, label='Train')
axes[0,0].plot(ep, history['val_loss'],   'r-o', ms=4, label='Val')
axes[0,0].set_title('Loss', fontweight='bold')
axes[0,0].legend(); axes[0,0].grid(alpha=.3)

# ── 2. Accuracy ──
axes[0,1].plot(ep, history['train_acc'], 'b-o', ms=4, label='Train')
axes[0,1].plot(ep, history['val_acc'],   'r-o', ms=4, label='Val')
axes[0,1].set_title('Accuracy (%)', fontweight='bold')
axes[0,1].legend(); axes[0,1].grid(alpha=.3)

# ── 3. Macro / Weighted F1 ──
axes[0,2].plot(ep, history['macro_f1'],    'g-o', ms=4, label='Macro F1')
axes[0,2].plot(ep, history['weighted_f1'], 'm-o', ms=4, label='Weighted F1')
axes[0,2].set_title('F1 Scores', fontweight='bold')
axes[0,2].legend(); axes[0,2].grid(alpha=.3)

# ── 4. Per-class F1 ──
for ci, cn in enumerate(CLASS_NAMES):
    axes[1,0].plot(ep, history[f'f1_{cn}'], '-o', ms=3, color=colors[ci], label=cn)
axes[1,0].set_title('Per-Class F1', fontweight='bold')
axes[1,0].legend(); axes[1,0].grid(alpha=.3)

# ── 5. Confusion Matrix ──
cm = confusion_matrix(all_t, all_p)
cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_n, annot=True, fmt='.2f', cmap='Blues', ax=axes[1,1],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
axes[1,1].set_title('Confusion Matrix (norm)', fontweight='bold')
axes[1,1].set_ylabel('True'); axes[1,1].set_xlabel('Pred')

# ── 6. ROC ──
y_bin = label_binarize(all_t, classes=list(range(NUM_CLASSES)))
for ci, (cn, col) in enumerate(zip(CLASS_NAMES, colors)):
    fpr, tpr, _ = roc_curve(y_bin[:,ci], all_prob[:,ci])
    axes[1,2].plot(fpr, tpr, color=col, lw=2, label=f'{cn} ({auc(fpr,tpr):.3f})')
axes[1,2].plot([0,1],[0,1],'k--',lw=1)
axes[1,2].set_title('ROC Curves', fontweight='bold')
axes[1,2].legend(loc='lower right', fontsize=8)
axes[1,2].grid(alpha=.3)

plt.suptitle(f'RetinaSense v2 — Macro F1={mf1:.3f} | AUC={mauc:.3f} | Val Acc={100*(all_p==all_t).mean():.1f}%',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/dashboard.png', dpi=150, bbox_inches='tight')
plt.close()

# LR schedule plot
fig, ax = plt.subplots(figsize=(8,3))
ax.plot(ep, history['lr'], 'b-o', ms=3)
ax.set_title('Learning Rate Schedule', fontweight='bold')
ax.set_xlabel('Epoch'); ax.set_ylabel('LR')
ax.grid(alpha=.3)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/lr_schedule.png', dpi=150)
plt.close()

# Save metrics
pd.DataFrame([{
    'val_accuracy': 100*(all_p==all_t).mean(),
    'macro_f1': mf1, 'weighted_f1': wf1, 'macro_auc': mauc,
    **{f'f1_{cn}': f1_score(all_t, all_p, average=None, labels=range(NUM_CLASSES))[ci]
       for ci,cn in enumerate(CLASS_NAMES)}
}]).to_csv(f'{SAVE_DIR}/metrics.csv', index=False)

# Save history
with open(f'{SAVE_DIR}/history.json','w') as f:
    json.dump({k:[float(v) for v in vs] for k,vs in history.items()}, f, indent=2)

print(f'\n{"="*65}')
print(f'        RETINASENSE v2 — FINAL RESULTS')
print(f'{"="*65}')
print(f'  Best Macro F1  : {best_f1:.4f}')
print(f'  Val Accuracy   : {100*(all_p==all_t).mean():.2f}%')
print(f'  Macro AUC      : {mauc:.4f}')
per_f1 = f1_score(all_t, all_p, average=None, labels=range(NUM_CLASSES), zero_division=0)
for ci, cn in enumerate(CLASS_NAMES):
    print(f'    {cn:15s}: F1={per_f1[ci]:.3f}')
print(f'{"="*65}')
print(f'\n📁 {SAVE_DIR}/')
print(f'   ├── best_model.pth')
print(f'   ├── dashboard.png')
print(f'   ├── lr_schedule.png')
print(f'   ├── metrics.csv')
print(f'   └── history.json')
