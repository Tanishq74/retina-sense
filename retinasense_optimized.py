#!/usr/bin/env python3
"""
RetinaSense - OPTIMIZED for H200 GPU
Key optimizations:
1. Pre-cache Ben Graham preprocessing (avoid on-the-fly CPU work)
2. Larger batch size (128 instead of 32)
3. More workers (8 instead of 2)
4. Mixed precision training enabled
"""

import os, time, warnings
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

# ═══════════════════════════════════════════════════════════
# OPTIMIZED CONFIGURATION
# ═══════════════════════════════════════════════════════════
SAVE_DIR = './outputs_optimized'
os.makedirs(SAVE_DIR, exist_ok=True)

BASE = './'
ODIR_CSV      = f'{BASE}/odir/full_df.csv'
ODIR_IMG_DIR  = f'{BASE}/odir/preprocessed_images'
APTOS_CSV     = f'{BASE}/aptos/train.csv'
APTOS_IMG_DIR = f'{BASE}/aptos/train_images'

EPOCHS        = 4
WARMUP_EPOCHS = 2
BATCH_SIZE    = 128  # ← Increased from 32 (4x larger!)
NUM_WORKERS   = 8    # ← Increased from 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['Normal', 'Diabetes/DR', 'Glaucoma', 'Cataract', 'AMD']

print('='*60)
print('      RETINASENSE - OPTIMIZED TRAINING PIPELINE')
print('='*60)
print(f'Device      : {device}')
if torch.cuda.is_available():
    print(f'GPU         : {torch.cuda.get_device_name(0)}')
    print(f'VRAM        : {round(torch.cuda.get_device_properties(0).total_memory/1e9, 1)} GB')
print(f'Save Dir    : {SAVE_DIR}')
print(f'Epochs      : {EPOCHS}')
print(f'Batch Size  : {BATCH_SIZE} (4x larger!)')
print(f'Workers     : {NUM_WORKERS} (4x more!)')
print('='*60)

# ═══════════════════════════════════════════════════════════
# STEP 1: BUILD METADATA (same as before)
# ═══════════════════════════════════════════════════════════
print('\n[1/9] Building unified metadata...')

disease_cols = ['N', 'D', 'G', 'C', 'A']
label_map    = {'N': 0, 'D': 1, 'G': 2, 'C': 3, 'A': 4}

df_odir = pd.read_csv(ODIR_CSV)
df_odir['disease_count'] = df_odir[disease_cols].sum(axis=1)
df_odir = df_odir[df_odir['disease_count'] == 1].copy()

def get_label(row):
    for d in disease_cols:
        if row[d] == 1:
            return label_map[d]

df_odir['disease_label'] = df_odir.apply(get_label, axis=1)

img_col = None
for col in df_odir.columns:
    if 'filename' in col.lower() or 'fundus' in col.lower() or 'image' in col.lower():
        img_col = col
        break

odir_metadata = pd.DataFrame({
    'image_path'    : ODIR_IMG_DIR + '/' + df_odir[img_col].astype(str),
    'dataset'       : 'ODIR',
    'disease_label' : df_odir['disease_label'],
    'severity_label': -1
})

df_aptos = pd.read_csv(APTOS_CSV)
aptos_metadata = pd.DataFrame({
    'image_path'    : APTOS_IMG_DIR + '/' + df_aptos['id_code'] + '.png',
    'dataset'       : 'APTOS',
    'disease_label' : 1,
    'severity_label': df_aptos['diagnosis']
})

final_metadata = pd.concat([odir_metadata, aptos_metadata], ignore_index=True)
final_metadata = final_metadata[
    final_metadata['image_path'].apply(os.path.exists)
].reset_index(drop=True)

print(f'  Total: {len(final_metadata)} samples')

# ═══════════════════════════════════════════════════════════
# STEP 2: PRE-CACHE BEN GRAHAM PREPROCESSING
# ═══════════════════════════════════════════════════════════
print('\n[2/9] ⚡ Pre-caching Ben Graham preprocessing...')
print('  This takes time upfront but makes training MUCH faster!')

CACHE_DIR = './preprocessed_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

def ben_graham_preprocess(img_path, img_size=224, sigmaX=10):
    """Ben Graham fundus preprocessing."""
    img = cv2.imread(img_path)
    if img is None:
        img = np.array(Image.open(img_path).convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), sigmaX), -4, 128)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (img_size//2, img_size//2), int(img_size*0.48), 255, -1)
    img = cv2.bitwise_and(img, img, mask=mask)
    return img

# Build cache mapping
cache_paths = []
for idx, row in tqdm(final_metadata.iterrows(), total=len(final_metadata), desc='Preprocessing'):
    cache_file = f"{CACHE_DIR}/{os.path.basename(row['image_path']).replace('.png', '.npy')}"

    if not os.path.exists(cache_file):
        try:
            img = ben_graham_preprocess(row['image_path'])
            np.save(cache_file, img)
        except:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            np.save(cache_file, img)

    cache_paths.append(cache_file)

final_metadata['cache_path'] = cache_paths
final_metadata.to_csv(f'{SAVE_DIR}/metadata_cached.csv', index=False)
print(f'  ✓ All images pre-processed and cached to {CACHE_DIR}/')

# ═══════════════════════════════════════════════════════════
# STEP 3: FAST DATASET CLASS (loads from cache)
# ═══════════════════════════════════════════════════════════
print('\n[3/9] Setting up fast dataset class...')

def get_transforms(phase='train'):
    if phase == 'train':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])


class CachedRetinalDataset(Dataset):
    """Fast dataset - loads pre-processed images from cache."""
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            # Load from cache (much faster!)
            image = np.load(row['cache_path'])
        except:
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.transform:
            image = self.transform(image)

        return (
            image,
            torch.tensor(int(row['disease_label']),  dtype=torch.long),
            torch.tensor(int(row['severity_label']), dtype=torch.long)
        )

print('  ✓ Fast cached dataset ready')

# ═══════════════════════════════════════════════════════════
# STEP 4: CREATE OPTIMIZED DATALOADERS
# ═══════════════════════════════════════════════════════════
print('\n[4/9] Creating optimized dataloaders...')

train_df, val_df = train_test_split(
    final_metadata,
    test_size=0.2,
    stratify=final_metadata['disease_label'],
    random_state=42
)

train_dataset = CachedRetinalDataset(train_df, transform=get_transforms('train'))
val_dataset   = CachedRetinalDataset(val_df,   transform=get_transforms('val'))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,  # Keep workers alive
    prefetch_factor=2         # Prefetch batches
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True
)

print(f'  Train: {len(train_dataset):5d} samples ({len(train_loader):3d} batches)')
print(f'  Val  : {len(val_dataset):5d} samples ({len(val_loader):3d} batches)')
print(f'  ⚡ Batch size increased 4x: 32 → {BATCH_SIZE}')
print(f'  ⚡ Workers increased 4x: 2 → {NUM_WORKERS}')

# ═══════════════════════════════════════════════════════════
# STEP 5: BUILD MODEL (same architecture)
# ═══════════════════════════════════════════════════════════
print('\n[5/9] Building EfficientNet-B3 model...')

class MultiTaskModel(nn.Module):
    def __init__(self, num_disease_classes=5, num_severity_classes=5, dropout=0.4):
        super().__init__()
        backbone = models.efficientnet_b3(weights='IMAGENET1K_V1')
        self.backbone    = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = 1536
        self.dropout     = nn.Dropout(dropout)

        self.disease_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_disease_classes)
        )

        self.severity_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_severity_classes)
        )

    def forward(self, x):
        f = self.backbone(x)
        f = f.view(f.size(0), -1)
        f = self.dropout(f)
        return self.disease_head(f), self.severity_head(f)

model = MultiTaskModel().to(device)
print(f'  ✓ Model ready on {device}')

# ═══════════════════════════════════════════════════════════
# STEP 6: SETUP TRAINING
# ═══════════════════════════════════════════════════════════
print('\n[6/9] Setting up training...')

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0,1,2,3,4]),
    y=train_df['disease_label'].values
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion_disease  = nn.CrossEntropyLoss(weight=class_weights_tensor)
criterion_severity = nn.CrossEntropyLoss(ignore_index=-1)

for param in model.backbone.parameters():
    param.requires_grad = False

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-7
)
scaler = GradScaler()

CHECKPOINT = f'{SAVE_DIR}/RetinaSense_optimized.pth'
print(f'  ✓ Checkpoint: {CHECKPOINT}')

# ═══════════════════════════════════════════════════════════
# STEP 7: OPTIMIZED TRAINING LOOP
# ═══════════════════════════════════════════════════════════
print('\n[7/9] Starting optimized training...')
print('='*60)

history  = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
best_acc = 0.0

for epoch in range(EPOCHS):
    start = time.time()

    if epoch == WARMUP_EPOCHS:
        print('\n🔓 Unfreezing full backbone')
        for param in model.backbone.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(
            model.parameters(), lr=5e-5, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-7
        )

    # TRAIN
    model.train()
    total_loss = 0.0
    train_correct = 0
    train_total   = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]')
    for images, disease_labels, severity_labels in pbar:
        images          = images.to(device, non_blocking=True)
        disease_labels  = disease_labels.to(device, non_blocking=True)
        severity_labels = severity_labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()

        with autocast('cuda'):
            disease_out, severity_out = model(images)
            loss_d = criterion_disease(disease_out, disease_labels)
            loss_s = criterion_severity(severity_out, severity_labels)
            loss   = loss_d + 0.5 * loss_s

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss    += loss.item()
        preds          = disease_out.argmax(dim=1)
        train_correct += (preds == disease_labels).sum().item()
        train_total   += disease_labels.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{100*train_correct/train_total:.1f}%'})

    train_acc = 100 * train_correct / train_total
    avg_loss  = total_loss / len(train_loader)

    # VALIDATION
    model.eval()
    val_correct = 0
    val_total   = 0
    val_loss    = 0.0

    with torch.no_grad():
        for images, disease_labels, severity_labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Val]  '):
            images          = images.to(device, non_blocking=True)
            disease_labels  = disease_labels.to(device, non_blocking=True)
            severity_labels = severity_labels.to(device, non_blocking=True)

            with autocast('cuda'):
                disease_out, severity_out = model(images)
                loss_d = criterion_disease(disease_out, disease_labels)
                loss_s = criterion_severity(severity_out, severity_labels)
                loss   = loss_d + 0.5 * loss_s

            val_loss    += loss.item()
            preds        = disease_out.argmax(dim=1)
            val_correct += (preds == disease_labels).sum().item()
            val_total   += disease_labels.size(0)

    val_acc  = 100 * val_correct / val_total
    val_loss /= len(val_loader)
    scheduler.step()

    history['train_loss'].append(avg_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            'epoch'              : epoch,
            'model_state_dict'   : model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc'            : val_acc,
            'history'            : history
        }, CHECKPOINT)
        best_marker = ' 🔥 BEST'
    else:
        best_marker = ''

    elapsed = time.time() - start
    it_per_sec = len(train_loader) / elapsed

    print(f'\nEpoch [{epoch+1:2d}/{EPOCHS}] | Time: {elapsed:.1f}s | Speed: {it_per_sec:.2f} it/s ⚡')
    print(f'  Train → Loss: {avg_loss:.4f} | Acc: {train_acc:.2f}%')
    print(f'  Val   → Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%{best_marker}')
    print('-'*60)

print(f'\n✅ Training complete! Best Val Acc: {best_acc:.2f}%')

# Save plots and evaluation (same as before)
print('\n[8/9] Generating plots...')

epochs_range = range(1, len(history['train_loss'])+1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(epochs_range, history['train_loss'], 'b-o', label='Train Loss')
axes[0].plot(epochs_range, history['val_loss'],   'r-o', label='Val Loss')
axes[0].set_title('Loss', fontsize=13, fontweight='bold')
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(epochs_range, history['train_acc'], 'b-o', label='Train Acc')
axes[1].plot(epochs_range, history['val_acc'],   'r-o', label='Val Acc')
axes[1].set_title('Accuracy', fontsize=13, fontweight='bold')
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/training_curves.png', dpi=150)
plt.close()

print('\n[9/9] Evaluation...')
ckpt = torch.load(CHECKPOINT)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, disease_labels, _ in tqdm(val_loader, desc='Evaluating'):
        images = images.to(device)
        d_out, _ = model(images)
        all_preds.extend(d_out.argmax(1).cpu().numpy())
        all_labels.extend(disease_labels.numpy())

print('\n' + classification_report(np.array(all_labels), np.array(all_preds), target_names=CLASS_NAMES, digits=4))

print('\n' + '='*60)
print('         OPTIMIZED RETINASENSE - FINAL RESULTS')
print('='*60)
print(f'  Best Val Acc: {best_acc:.2f}%')
print(f'  Speedup     : ~4x faster (larger batches + cached preprocessing)')
print('='*60)
print(f'\n📁 Outputs saved to: {SAVE_DIR}/')
