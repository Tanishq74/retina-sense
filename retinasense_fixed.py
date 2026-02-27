#!/usr/bin/env python3
"""
RetinaSense - Multi-Task Retinal Disease Classification
EfficientNet-B3 with Ben Graham Preprocessing
"""

import os, time, warnings
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
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
# CONFIGURATION
# ═══════════════════════════════════════════════════════════
SAVE_DIR = './outputs'
os.makedirs(SAVE_DIR, exist_ok=True)

BASE = './'
ODIR_CSV      = f'{BASE}/odir/full_df.csv'
ODIR_IMG_DIR  = f'{BASE}/odir/preprocessed_images'
APTOS_CSV     = f'{BASE}/aptos/train.csv'
APTOS_IMG_DIR = f'{BASE}/aptos/train_images'

EPOCHS        = 4   # Change to 50 for full training
WARMUP_EPOCHS = 2
BATCH_SIZE    = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['Normal', 'Diabetes/DR', 'Glaucoma', 'Cataract', 'AMD']

print('='*60)
print('           RETINASENSE - TRAINING PIPELINE')
print('='*60)
print(f'Device      : {device}')
if torch.cuda.is_available():
    print(f'GPU         : {torch.cuda.get_device_name(0)}')
    print(f'VRAM        : {round(torch.cuda.get_device_properties(0).total_memory/1e9, 1)} GB')
print(f'Save Dir    : {SAVE_DIR}')
print(f'Epochs      : {EPOCHS}')
print('='*60)

# ═══════════════════════════════════════════════════════════
# STEP 1: BUILD UNIFIED METADATA
# ═══════════════════════════════════════════════════════════
print('\n[1/8] Building unified metadata...')

disease_cols = ['N', 'D', 'G', 'C', 'A']
label_map    = {'N': 0, 'D': 1, 'G': 2, 'C': 3, 'A': 4}

# ODIR
df_odir = pd.read_csv(ODIR_CSV)
df_odir['disease_count'] = df_odir[disease_cols].sum(axis=1)
df_odir = df_odir[df_odir['disease_count'] == 1].copy()

def get_label(row):
    for d in disease_cols:
        if row[d] == 1:
            return label_map[d]

df_odir['disease_label'] = df_odir.apply(get_label, axis=1)

# Auto-detect image column
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

# APTOS
df_aptos = pd.read_csv(APTOS_CSV)
aptos_metadata = pd.DataFrame({
    'image_path'    : APTOS_IMG_DIR + '/' + df_aptos['id_code'] + '.png',
    'dataset'       : 'APTOS',
    'disease_label' : 1,
    'severity_label': df_aptos['diagnosis']
})

# Merge & clean
final_metadata = pd.concat([odir_metadata, aptos_metadata], ignore_index=True)
final_metadata = final_metadata[
    final_metadata['image_path'].apply(os.path.exists)
].reset_index(drop=True)

final_metadata.to_csv(f'{SAVE_DIR}/metadata.csv', index=False)

print(f'  ODIR  : {len(odir_metadata)} samples')
print(f'  APTOS : {len(aptos_metadata)} samples')
print(f'  Total : {len(final_metadata)} samples (after path check)')
print('\n  Class distribution:')
for i, cnt in final_metadata['disease_label'].value_counts().sort_index().items():
    print(f'    {CLASS_NAMES[i]:15s}: {cnt:4d}')

# ═══════════════════════════════════════════════════════════
# STEP 2: BEN GRAHAM PREPROCESSING
# ═══════════════════════════════════════════════════════════
print('\n[2/8] Setting up preprocessing...')

def ben_graham_preprocess(img_path, img_size=224, sigmaX=10):
    """Ben Graham fundus preprocessing - enhances retinal structures."""
    img = cv2.imread(img_path)
    if img is None:
        img = np.array(Image.open(img_path).convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.addWeighted(img, 4,
                          cv2.GaussianBlur(img, (0,0), sigmaX), -4, 128)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (img_size//2, img_size//2), int(img_size*0.48), 255, -1)
    img = cv2.bitwise_and(img, img, mask=mask)
    return Image.fromarray(img)


def get_transforms(phase='train'):
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])


class RetinalDataset(Dataset):
    def __init__(self, dataframe, transform=None, use_ben_graham=True):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.use_ben_graham = use_ben_graham

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            image = ben_graham_preprocess(row['image_path']) if self.use_ben_graham \
                    else Image.open(row['image_path']).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), (128, 128, 128))
        if self.transform:
            image = self.transform(image)
        return (
            image,
            torch.tensor(int(row['disease_label']),  dtype=torch.long),
            torch.tensor(int(row['severity_label']), dtype=torch.long)
        )

print('  ✓ Ben Graham preprocessing ready')

# Visualize preprocessing
try:
    sample = final_metadata[final_metadata['dataset']=='APTOS'].iloc[0]
    orig   = Image.open(sample['image_path']).convert('RGB').resize((224,224))
    proc   = ben_graham_preprocess(sample['image_path'])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(orig);  axes[0].set_title('Original Fundus',         fontsize=13); axes[0].axis('off')
    axes[1].imshow(proc);  axes[1].set_title('Ben Graham Preprocessed', fontsize=13); axes[1].axis('off')
    plt.suptitle('Preprocessing Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/preprocessing.png', dpi=150)
    plt.close()
    print('  ✓ Preprocessing comparison saved')
except Exception as e:
    print(f'  ⚠ Preprocessing visualization skipped: {e}')

# ═══════════════════════════════════════════════════════════
# STEP 3: CREATE DATALOADERS
# ═══════════════════════════════════════════════════════════
print('\n[3/8] Creating dataloaders...')

train_df, val_df = train_test_split(
    final_metadata,
    test_size=0.2,
    stratify=final_metadata['disease_label'],
    random_state=42
)

train_dataset = RetinalDataset(train_df, transform=get_transforms('train'), use_ben_graham=True)
val_dataset   = RetinalDataset(val_df,   transform=get_transforms('val'),   use_ben_graham=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)

print(f'  Train: {len(train_dataset):5d} samples ({len(train_loader):3d} batches)')
print(f'  Val  : {len(val_dataset):5d} samples ({len(val_loader):3d} batches)')

# Class distribution plot
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
colors = ['#2ecc71','#3498db','#e74c3c','#f39c12','#9b59b6']
for ax, df, title in zip(axes, [train_df, val_df], ['Train', 'Validation']):
    counts = df['disease_label'].value_counts().sort_index()
    bars = ax.bar([CLASS_NAMES[i] for i in counts.index], counts.values, color=colors)
    ax.set_title(f'{title} Set Distribution', fontsize=13, fontweight='bold')
    ax.set_ylabel('Count')
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x()+bar.get_width()/2, v+5, str(v), ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/class_distribution.png', dpi=150)
plt.close()
print('  ✓ Class distribution plot saved')

# ═══════════════════════════════════════════════════════════
# STEP 4: BUILD MODEL
# ═══════════════════════════════════════════════════════════
print('\n[4/8] Building EfficientNet-B3 model...')

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

total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'  Total params    : {total:,}')
print(f'  Trainable params: {trainable:,}')

# ═══════════════════════════════════════════════════════════
# STEP 5: SETUP TRAINING
# ═══════════════════════════════════════════════════════════
print('\n[5/8] Setting up training...')

# Auto class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0,1,2,3,4]),
    y=train_df['disease_label'].values
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
print('  Class weights:')
for name, w in zip(CLASS_NAMES, class_weights):
    print(f'    {name:15s}: {w:.3f}')

criterion_disease  = nn.CrossEntropyLoss(weight=class_weights_tensor)
criterion_severity = nn.CrossEntropyLoss(ignore_index=-1)

# Phase 1: freeze backbone
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

CHECKPOINT = f'{SAVE_DIR}/RetinaSense_best_model.pth'
print(f'  ✓ Checkpoint: {CHECKPOINT}')

# ═══════════════════════════════════════════════════════════
# STEP 6: TRAINING LOOP
# ═══════════════════════════════════════════════════════════
print('\n[6/8] Starting training...')
print('='*60)

history  = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
best_acc = 0.0

for epoch in range(EPOCHS):
    start = time.time()

    # Unfreeze backbone after warmup
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

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]', leave=False)
    for batch_idx, (images, disease_labels, severity_labels) in enumerate(pbar):
        images          = images.to(device)
        disease_labels  = disease_labels.to(device)
        severity_labels = severity_labels.to(device)

        optimizer.zero_grad()

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

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    train_acc = 100 * train_correct / train_total
    avg_loss  = total_loss / len(train_loader)

    # VALIDATION
    model.eval()
    val_correct = 0
    val_total   = 0
    val_loss    = 0.0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Val]  ', leave=False)
        for images, disease_labels, severity_labels in pbar:
            images          = images.to(device)
            disease_labels  = disease_labels.to(device)
            severity_labels = severity_labels.to(device)

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

    lr = optimizer.param_groups[0]['lr']

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
    print(f'\nEpoch [{epoch+1:2d}/{EPOCHS}] | LR: {lr:.2e} | Time: {elapsed:.1f}s')
    print(f'  Train → Loss: {avg_loss:.4f} | Acc: {train_acc:.2f}%')
    print(f'  Val   → Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%{best_marker}')
    print('-'*60)

print(f'\n✅ Training complete! Best Val Acc: {best_acc:.2f}%')

# ═══════════════════════════════════════════════════════════
# STEP 7: TRAINING CURVES
# ═══════════════════════════════════════════════════════════
print('\n[7/8] Generating training curves...')

epochs_range = range(1, len(history['train_loss'])+1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(epochs_range, history['train_loss'], 'b-o', label='Train Loss')
axes[0].plot(epochs_range, history['val_loss'],   'r-o', label='Val Loss')
axes[0].set_title('Loss Curve', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(epochs_range, history['train_acc'], 'b-o', label='Train Acc')
axes[1].plot(epochs_range, history['val_acc'],   'r-o', label='Val Acc')
axes[1].set_title('Accuracy Curve', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.suptitle('RetinaSense — EfficientNet-B3 Training', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/training_curves.png', dpi=150)
plt.close()
print('  ✓ Training curves saved')

# ═══════════════════════════════════════════════════════════
# STEP 8: FULL EVALUATION
# ═══════════════════════════════════════════════════════════
print('\n[8/8] Running full evaluation...')

# Load best model
ckpt = torch.load(CHECKPOINT, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f'  ✓ Loaded best model (epoch {ckpt["epoch"]+1}, val_acc={ckpt["val_acc"]:.2f}%)')

all_preds, all_labels, all_probs = [], [], []

with torch.no_grad():
    for images, disease_labels, _ in tqdm(val_loader, desc='Evaluating'):
        images = images.to(device)
        d_out, _ = model(images)
        probs = torch.softmax(d_out, dim=1)
        all_preds.extend(d_out.argmax(1).cpu().numpy())
        all_labels.extend(disease_labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

# Classification Report
print('\n' + '='*60)
print('               CLASSIFICATION REPORT')
print('='*60)
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4))

macro_f1    = f1_score(all_labels, all_preds, average='macro')
weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
print(f'Macro F1    : {macro_f1:.4f}')
print(f'Weighted F1 : {weighted_f1:.4f}')

try:
    macro_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    print(f'Macro AUC   : {macro_auc:.4f}')
except Exception as e:
    macro_auc = 'N/A'
    print(f'AUC skipped : {e}')

# Save metrics
pd.DataFrame([{
    'val_accuracy': ckpt['val_acc'],
    'macro_f1'    : macro_f1,
    'weighted_f1' : weighted_f1,
    'macro_auc'   : macro_auc
}]).to_csv(f'{SAVE_DIR}/metrics_summary.csv', index=False)

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(cm,      annot=True, fmt='d',   cmap='Blues',  ax=axes[0],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, linewidths=0.5)
axes[0].set_title('Confusion Matrix (Counts)',     fontsize=13, fontweight='bold')
axes[0].set_ylabel('True'); axes[0].set_xlabel('Predicted')
axes[0].tick_params(axis='x', rotation=30)

sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens', ax=axes[1],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, vmin=0, vmax=1)
axes[1].set_title('Confusion Matrix (Normalized)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('True'); axes[1].set_xlabel('Predicted')
axes[1].tick_params(axis='x', rotation=30)

plt.suptitle('RetinaSense — Disease Classification Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print('  ✓ Confusion matrix saved')

# ROC Curves
y_bin  = label_binarize(all_labels, classes=[0,1,2,3,4])
colors = ['#2ecc71','#3498db','#e74c3c','#f39c12','#9b59b6']

plt.figure(figsize=(9, 7))
for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
    fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'{name} (AUC = {auc(fpr, tpr):.3f})')

plt.plot([0,1],[0,1],'k--', lw=1.5, label='Random')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate',  fontsize=12)
plt.title('ROC Curves — Per Disease Class', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}/roc_curves.png', dpi=150)
plt.close()
print('  ✓ ROC curves saved')

# ═══════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════
print('\n' + '='*60)
print('            RETINASENSE — FINAL RESULTS')
print('='*60)
print(f'  Backbone       : EfficientNet-B3')
print(f'  Total Samples  : {len(final_metadata)}')
print(f'  Train/Val Split: {len(train_dataset)}/{len(val_dataset)}')
print(f'  Epochs Trained : {EPOCHS}')
print(f'  Best Val Acc   : {ckpt["val_acc"]:.2f}%')
print(f'  Macro F1       : {macro_f1:.4f}')
print(f'  Weighted F1    : {weighted_f1:.4f}')
try:
    print(f'  Macro AUC      : {macro_auc:.4f}')
except: pass
print('='*60)
print(f'\n📁 All outputs saved to: {SAVE_DIR}/')
print('   ├── RetinaSense_best_model.pth')
print('   ├── metadata.csv')
print('   ├── preprocessing.png')
print('   ├── class_distribution.png')
print('   ├── training_curves.png')
print('   ├── confusion_matrix.png')
print('   ├── roc_curves.png')
print('   └── metrics_summary.csv')
print('\n✅ Pipeline complete!')
print('\n💡 To train for full performance, set EPOCHS=50 at the top of the script')
