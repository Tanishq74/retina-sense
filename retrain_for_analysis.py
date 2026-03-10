#!/usr/bin/env python3
"""
Retrain RetinaSense ViT to reproduce best_model.pth for error analysis.
Uses exact same config as retinasense_vit.py but reads from the correct data paths.
Pre-caches all images first for fast training on H100.
"""

import os, sys, time, warnings, json
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report

# ================================================================
# CONFIG — identical to retinasense_vit.py
# ================================================================
BASE_DIR      = '/teamspace/studios/this_studio'
SAVE_DIR      = f'{BASE_DIR}/outputs_vit'
CACHE_DIR     = f'{BASE_DIR}/preprocessed_cache_vit'
META_CSV      = f'{BASE_DIR}/final_unified_metadata.csv'

EPOCHS        = 30
WARMUP_EPOCHS = 3
LR_WARMUP_STEPS = 3
BATCH_SIZE    = 32
ACCUM_STEPS   = 2
NUM_WORKERS   = 8
PATIENCE      = 10
FOCAL_GAMMA   = 1.0
IMG_SIZE      = 224

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['Normal', 'Diabetes/DR', 'Glaucoma', 'Cataract', 'AMD']
NUM_CLASSES = 5

print('='*65)
print('     RetinaSense ViT Retraining (for error analysis)')
print('='*65)
if torch.cuda.is_available():
    print(f'  GPU : {torch.cuda.get_device_name(0)}')
print(f'  Device : {device}')
print(f'  IMG_SIZE={IMG_SIZE}, BATCH={BATCH_SIZE}, EPOCHS={EPOCHS}')
print('='*65)

# ================================================================
# METADATA
# ================================================================
print('\n[1/6] Loading metadata...')
meta = pd.read_csv(META_CSV)
print(f'  Rows: {len(meta)}')

def resolve_path(raw_path):
    """Resolve paths with leading .// prefix."""
    clean = raw_path.lstrip('.').lstrip('/').replace('//', '/')
    stem = Path(raw_path).stem

    candidates = [f'{BASE_DIR}/{clean}']

    if 'aptos' in raw_path.lower():
        aptos_base = f'{BASE_DIR}/aptos/gaussian_filtered_images/gaussian_filtered_images'
        for sev in ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR']:
            for ext in ['.png', '.jpg', '.jpeg']:
                candidates.append(f'{aptos_base}/{sev}/{stem}{ext}')

    if 'odir' in raw_path.lower():
        fname = Path(raw_path).name
        candidates.append(f'{BASE_DIR}/odir/preprocessed_images/{fname}')

    for c in candidates:
        if os.path.exists(c):
            return c
    return None

meta['image_path_resolved'] = meta['image_path'].apply(resolve_path)
meta = meta[meta['image_path_resolved'].notna()].reset_index(drop=True)
print(f'  Valid rows: {len(meta)}')
for lbl, cnt in meta['disease_label'].value_counts().sort_index().items():
    print(f'    {CLASS_NAMES[int(lbl)]:<15s}: {cnt:4d}')

# ================================================================
# PRE-CACHE (Ben Graham preprocessing)
# ================================================================
print(f'\n[2/6] Pre-caching images @ {IMG_SIZE}x{IMG_SIZE}...')

def ben_graham(path, sz=IMG_SIZE, sigma=10):
    img = cv2.imread(str(path))
    if img is None:
        img = np.array(Image.open(str(path)).convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (sz, sz))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), sigma), -4, 128)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (sz//2, sz//2), int(sz*0.48), 255, -1)
    return cv2.bitwise_and(img, img, mask=mask)

cache_paths = []
cached = 0
for _, row in tqdm(meta.iterrows(), total=len(meta), desc='Caching'):
    stem = Path(row['image_path_resolved']).stem
    fp   = f'{CACHE_DIR}/{stem}_{IMG_SIZE}.npy'
    if not os.path.exists(fp):
        try:
            np.save(fp, ben_graham(row['image_path_resolved']))
        except Exception:
            np.save(fp, np.zeros((IMG_SIZE,IMG_SIZE,3), dtype=np.uint8))
        cached += 1
    cache_paths.append(fp)

meta['cache_path'] = cache_paths
print(f'  Newly cached: {cached} | Already cached: {len(meta)-cached}')

# ================================================================
# DATASET
# ================================================================
print('\n[3/6] Creating data loaders...')

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

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=True, prefetch_factor=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=True)

print(f'  Train : {len(train_ds):5d}  ({len(train_loader):3d} batches)')
print(f'  Val   : {len(val_ds):5d}  ({len(val_loader):3d} batches)')

# ================================================================
# MODEL + FOCAL LOSS
# ================================================================
print('\n[4/6] Building ViT model...')

class FocalLoss(nn.Module):
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

class MultiTaskViT(nn.Module):
    def __init__(self, n_disease=5, n_severity=5, drop=0.4):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        feat = 768
        self.drop = nn.Dropout(drop)
        self.disease_head = nn.Sequential(
            nn.Linear(feat, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_disease))
        self.severity_head = nn.Sequential(
            nn.Linear(feat, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_severity))
    def forward(self, x):
        f = self.backbone(x)
        f = self.drop(f)
        return self.disease_head(f), self.severity_head(f)

model = MultiTaskViT().to(device)

cw = compute_class_weight('balanced', classes=np.arange(5), y=train_df['disease_label'].values)
alpha = torch.tensor(cw, dtype=torch.float32).to(device)
alpha = alpha / alpha.sum() * NUM_CLASSES
print(f'  Focal alpha: {[f"{a:.2f}" for a in alpha.tolist()]}')

criterion_d = FocalLoss(alpha=alpha, gamma=FOCAL_GAMMA)
criterion_s = nn.CrossEntropyLoss(ignore_index=-1)

total_p = sum(p.numel() for p in model.parameters())
print(f'  Params: {total_p:,}')

# ================================================================
# TRAINING LOOP
# ================================================================
print('\n[5/6] Training...')

for p in model.backbone.parameters():
    p.requires_grad = False

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=3e-4, weight_decay=1e-3)
scaler = GradScaler()

def get_scheduler(opt, warmup_steps, total_steps):
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
sched = get_scheduler(optimizer, warmup_steps=len(train_loader)//ACCUM_STEPS,
                      total_steps=total_steps)

t_start = time.time()
print('='*65)

for epoch in range(EPOCHS):
    t0 = time.time()

    if epoch == WARMUP_EPOCHS:
        print('\n  Unfreezing ViT backbone')
        for p in model.backbone.parameters():
            p.requires_grad = True
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

    # TRAIN
    model.train()
    run_loss = 0.0; correct = 0; total = 0
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
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(set_to_none=True); continue
        scaler.scale(loss).backward()
        if (step + 1) % ACCUM_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            sched.step()
        run_loss += loss.item() * ACCUM_STEPS
        preds    = d_out.argmax(1)
        correct += (preds == d_lbl).sum().item()
        total   += d_lbl.size(0)
        pbar.set_postfix(loss=f'{loss.item()*ACCUM_STEPS:.3f}', acc=f'{100*correct/total:.1f}%')

    train_loss = run_loss / len(train_loader)
    train_acc  = 100 * correct / total

    # VALIDATE
    model.eval()
    vl = 0.0
    all_p, all_t, all_prob = [], [], []
    with torch.no_grad():
        for imgs, d_lbl, s_lbl in tqdm(val_loader, desc=f'E{epoch+1:02d}/{EPOCHS} val', leave=False):
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
        tag = f' * NEW BEST'
    else:
        patience_ctr += 1

    cls_str = ' | '.join(f'{cn[:3]}:{per_f1[ci]:.3f}' for ci,cn in enumerate(CLASS_NAMES))
    print(f'E{epoch+1:02d}/{EPOCHS} | {elapsed:.0f}s | LR {lr:.1e} | '
          f'TrL {train_loss:.3f} TrA {train_acc:.1f}% | '
          f'VL {val_loss:.3f} VA {val_acc:.1f}% | '
          f'mF1 {mf1:.4f}{tag}')
    print(f'       {cls_str}')

    if patience_ctr >= PATIENCE:
        print(f'\n  Early stopping at epoch {epoch+1}')
        break

total_time = time.time() - t_start
print(f'\nTraining done in {total_time/60:.1f} min. Best macro-F1: {best_f1:.4f}')
print(f'Checkpoint saved to: {CHECKPOINT}')

# Save history
with open(f'{SAVE_DIR}/history.json', 'w') as f:
    json.dump({k:[float(v) for v in vs] for k,vs in history.items()}, f, indent=2)
print('Done. Run run_error_analysis.py next.')
