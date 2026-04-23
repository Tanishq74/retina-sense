#!/usr/bin/env python3
"""
RetinaSense v3.0 — Phase 2B: EfficientNet-B3 Ensemble Training
================================================================
Trains EfficientNet-B3 as second backbone using identical training recipe,
then optimizes ensemble weights on calibration set.
"""

import os, sys, json, time, warnings, numpy as np, pandas as pd, cv2
from PIL import Image
from tqdm import tqdm
from collections import Counter
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
from scipy.optimize import minimize_scalar
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize

# ================================================================
# CONFIG
# ================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224
NUM_CLASSES = 5
CLASS_NAMES = ['Normal', 'Diabetes/DR', 'Glaucoma', 'Cataract', 'AMD']
BATCH_SIZE = 32
NUM_EPOCHS = 30
PATIENCE = 8
BASE_LR = 1e-3
WEIGHT_DECAY = 1e-4
MIXUP_ALPHA = 0.4
FOCAL_GAMMA = 1.0
GRAD_ACCUM = 2
NUM_WORKERS = 8
CACHE_DIR = './preprocessed_cache_unified'
OUTPUT_DIR = './outputs_v3/ensemble'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load norm stats
_norm_path = './configs/fundus_norm_stats_unified.json'
if not os.path.exists(_norm_path):
    _norm_path = './data/fundus_norm_stats.json'
with open(_norm_path) as f:
    ns = json.load(f)
NORM_MEAN, NORM_STD = ns['mean_rgb'], ns['std_rgb']

print('=' * 65)
print('  RetinaSense v3.0 — EfficientNet-B3 Ensemble Training')
print('=' * 65)
print(f'  Device: {DEVICE}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')

# ================================================================
# DATA LOADING (reuse existing splits and cache)
# ================================================================
train_df = pd.read_csv('./data/train_split.csv')
calib_df = pd.read_csv('./data/calib_split.csv')
test_df  = pd.read_csv('./data/test_split.csv')
print(f'\n  Train: {len(train_df)} | Calib: {len(calib_df)} | Test: {len(test_df)}')


def _cache_key(image_path):
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(CACHE_DIR, f'{stem}_{IMG_SIZE}.npy')


def make_transforms(phase):
    normalize = transforms.Normalize(NORM_MEAN, NORM_STD)
    if phase == 'train':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(20),
            transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.2),
        ])
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        normalize,
    ])


class RetinalDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cache_fp = row.get('cache_path', _cache_key(row['image_path']))
        # Prefer unified cache
        if isinstance(cache_fp, str):
            unified = cache_fp.replace('preprocessed_cache_v3', 'preprocessed_cache_unified')
            if os.path.exists(unified):
                cache_fp = unified
            elif not os.path.isabs(cache_fp):
                cache_fp = os.path.join('.', cache_fp.lstrip('./'))
        try:
            img = np.load(cache_fp)
        except Exception:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        img_tensor = self.transform(img)
        disease_lbl = int(row['disease_label'])
        return img_tensor, torch.tensor(disease_lbl, dtype=torch.long)


def _make_weighted_sampler(df):
    labels = df['disease_label'].values
    class_cnt = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    class_cnt = np.where(class_cnt == 0, 1.0, class_cnt)
    weights = 1.0 / class_cnt[labels]
    return WeightedRandomSampler(torch.DoubleTensor(weights), len(weights), replacement=True)


train_ds = RetinalDataset(train_df, make_transforms('train'))
calib_ds = RetinalDataset(calib_df, make_transforms('val'))
test_ds  = RetinalDataset(test_df,  make_transforms('val'))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=_make_weighted_sampler(train_df),
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=2)
calib_loader = DataLoader(calib_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)


# ================================================================
# EFFICIENTNET-B3 MODEL
# ================================================================
class EfficientNetB3(nn.Module):
    def __init__(self, n_classes=NUM_CLASSES, drop=0.3):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
        feat_dim = self.backbone.num_features  # 1536 for B3
        self.drop = nn.Dropout(drop)
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        f = self.backbone(x)
        f = self.drop(f)
        return self.head(f)

    def get_features(self, x):
        return self.backbone(x)


print('\n  Building EfficientNet-B3...')
model = EfficientNetB3().to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f'  Parameters: {total_params:,}')


# ================================================================
# FOCAL LOSS
# ================================================================
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


cw = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=train_df['disease_label'].values)
alpha = torch.tensor(cw, dtype=torch.float32).to(DEVICE)
alpha = alpha / alpha.sum() * NUM_CLASSES
criterion = FocalLoss(alpha=alpha, gamma=FOCAL_GAMMA)


# ================================================================
# OPTIMIZER + SCHEDULER
# ================================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=BASE_LR,
    steps_per_epoch=len(train_loader), epochs=NUM_EPOCHS,
    pct_start=0.1, anneal_strategy='cos', div_factor=10.0, final_div_factor=100.0
)
scaler = GradScaler()


# ================================================================
# MIXUP
# ================================================================
def mixup_data(x, y, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam


# ================================================================
# TRAINING
# ================================================================
print(f'\n  Training EfficientNet-B3 for {NUM_EPOCHS} epochs...')

CHECKPOINT = os.path.join(OUTPUT_DIR, 'efficientnet_b3.pth')
best_f1 = 0.0
patience_ctr = 0
t_start = time.time()

for epoch in range(NUM_EPOCHS):
    t0 = time.time()
    model.train()
    run_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(train_loader, desc=f'E{epoch+1:02d}/{NUM_EPOCHS} train', leave=False)
    for step, (imgs, labels) in enumerate(pbar):
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        mixed_imgs, y_a, y_b, lam = mixup_data(imgs, labels)
        with autocast('cuda'):
            logits = model(mixed_imgs)
            loss = (lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)) / GRAD_ACCUM

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        if (step + 1) % GRAD_ACCUM == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        run_loss += loss.item() * GRAD_ACCUM
        with torch.no_grad():
            correct += (logits.argmax(1) == y_a).sum().item()
            total += labels.size(0)
        pbar.set_postfix(loss=f'{loss.item()*GRAD_ACCUM:.3f}', acc=f'{100*correct/total:.1f}%')

    # Flush remaining
    if len(train_loader) % GRAD_ACCUM != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    train_loss = run_loss / len(train_loader)
    train_acc = 100 * correct / total

    # Validate
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    val_loss = 0.0
    with torch.no_grad():
        for imgs, labels in tqdm(calib_loader, desc='val', leave=False):
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            with autocast('cuda'):
                logits = model(imgs)
                loss = criterion(logits, labels)
            val_loss += loss.item()
            probs = torch.softmax(logits.float(), dim=1)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    val_loss /= len(calib_loader)
    preds = np.array(all_preds)
    targets = np.array(all_targets)
    val_acc = 100 * (preds == targets).mean()
    mf1 = f1_score(targets, preds, average='macro')
    per_f1 = f1_score(targets, preds, average=None, labels=range(NUM_CLASSES), zero_division=0)

    elapsed = time.time() - t0
    tag = ''
    if mf1 > best_f1 + 0.001:
        best_f1 = mf1
        patience_ctr = 0
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                     'val_acc': val_acc, 'macro_f1': mf1}, CHECKPOINT)
        tag = f'  * BEST (F1={mf1:.4f})'
    else:
        patience_ctr += 1

    cls_str = ' | '.join(f'{cn[:3]}:{per_f1[ci]:.2f}' for ci, cn in enumerate(CLASS_NAMES))
    print(f'E{epoch+1:02d} | {elapsed:.0f}s | TrL {train_loss:.3f} TrA {train_acc:.1f}% | '
          f'VL {val_loss:.3f} VA {val_acc:.1f}% | mF1 {mf1:.4f}{tag}')
    print(f'     {cls_str}')

    if patience_ctr >= PATIENCE:
        print(f'\n  Early stopping at epoch {epoch+1}')
        break

train_time = time.time() - t_start
print(f'\nEfficientNet-B3 training done. Best F1: {best_f1:.4f} ({train_time/60:.1f} min)')


# ================================================================
# ENSEMBLE OPTIMIZATION
# ================================================================
print('\n' + '=' * 65)
print('  Optimizing Ensemble Weights')
print('=' * 65)

# Load ViT model
sys.path.insert(0, '.')

# Rebuild ViT architecture
class MultiTaskViT(nn.Module):
    def __init__(self, n_disease=5, n_severity=5, drop=0.3):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
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


vit_model = MultiTaskViT().to(DEVICE)
_vit_path = './outputs_v3/dann/best_model.pth'
if not os.path.exists(_vit_path):
    _vit_path = './outputs_v3/best_model.pth'
ckpt = torch.load(_vit_path, map_location=DEVICE, weights_only=False)
_sd = ckpt['model_state_dict']
_filtered = {k: v for k, v in _sd.items()
             if not k.startswith('domain_head') and not k.startswith('grl')}
vit_model.load_state_dict(_filtered, strict=False)
vit_model.eval()

# Load best EfficientNet
ckpt_eff = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt_eff['model_state_dict'])
model.eval()

# Load temperature
_temp_path = './configs/temperature.json'
if not os.path.exists(_temp_path):
    _temp_path = './outputs_v3/temperature.json'
with open(_temp_path) as f:
    T_opt = json.load(f)['temperature']

print(f'  ViT checkpoint: epoch {ckpt.get("epoch",0)+1}, F1={ckpt.get("macro_f1",0):.4f}')
print(f'  EfficientNet checkpoint: epoch {ckpt_eff["epoch"]+1}, F1={ckpt_eff["macro_f1"]:.4f}')
print(f'  Temperature: {T_opt:.4f}')


# Collect predictions from both models on calib set
def collect_probs(mdl, loader, is_vit=False):
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Collecting', leave=False):
            imgs = batch[0].to(DEVICE, non_blocking=True)
            labels = batch[1].to(DEVICE, non_blocking=True)
            with autocast('cuda'):
                if is_vit:
                    logits, _ = mdl(imgs)
                else:
                    logits = mdl(imgs)
            probs = torch.softmax(logits.float() / T_opt, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


print('\n  Collecting ViT predictions on calib set...')
vit_probs_calib, calib_labels = collect_probs(vit_model, calib_loader, is_vit=True)

print('  Collecting EfficientNet predictions on calib set...')
eff_probs_calib, _ = collect_probs(model, calib_loader, is_vit=False)

# Grid search ensemble weight
print('\n  Grid searching ensemble weight...')
best_w, best_ens_f1 = 0.5, 0.0
for w_vit in np.arange(0.1, 0.95, 0.05):
    ens_probs = w_vit * vit_probs_calib + (1 - w_vit) * eff_probs_calib
    ens_preds = ens_probs.argmax(axis=1)
    mf1 = f1_score(calib_labels, ens_preds, average='macro')
    if mf1 > best_ens_f1:
        best_ens_f1 = mf1
        best_w = w_vit

print(f'  Best weight: ViT={best_w:.2f}, EfficientNet={1-best_w:.2f}')
print(f'  Ensemble calib F1: {best_ens_f1:.4f}')
print(f'  ViT-only calib F1: {f1_score(calib_labels, vit_probs_calib.argmax(1), average="macro"):.4f}')
print(f'  Eff-only calib F1: {f1_score(calib_labels, eff_probs_calib.argmax(1), average="macro"):.4f}')


# ================================================================
# FINAL EVALUATION ON TEST SET
# ================================================================
print('\n' + '=' * 65)
print('  ENSEMBLE TEST SET EVALUATION')
print('=' * 65)

print('  Collecting test predictions...')
vit_probs_test, test_labels = collect_probs(vit_model, test_loader, is_vit=True)
eff_probs_test, _ = collect_probs(model, test_loader, is_vit=False)

ens_probs_test = best_w * vit_probs_test + (1 - best_w) * eff_probs_test
ens_preds_test = ens_probs_test.argmax(axis=1)

# Disagreement analysis
vit_preds = vit_probs_test.argmax(axis=1)
eff_preds = eff_probs_test.argmax(axis=1)
disagree_mask = vit_preds != eff_preds
n_disagree = disagree_mask.sum()

print(f'\n  Model Disagreement: {n_disagree}/{len(test_labels)} ({100*n_disagree/len(test_labels):.1f}%)')
if n_disagree > 0:
    disagree_acc_vit = (vit_preds[disagree_mask] == test_labels[disagree_mask]).mean()
    disagree_acc_eff = (eff_preds[disagree_mask] == test_labels[disagree_mask]).mean()
    disagree_acc_ens = (ens_preds_test[disagree_mask] == test_labels[disagree_mask]).mean()
    print(f'  On disagreed samples:')
    print(f'    ViT accuracy:      {100*disagree_acc_vit:.1f}%')
    print(f'    EfficientNet acc:  {100*disagree_acc_eff:.1f}%')
    print(f'    Ensemble accuracy: {100*disagree_acc_ens:.1f}%')


def print_metrics(name, preds, labels, probs):
    acc = 100 * (preds == labels).mean()
    mf1 = f1_score(labels, preds, average='macro')
    wf1 = f1_score(labels, preds, average='weighted')
    try:
        mauc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    except:
        mauc = 0.0
    print(f'\n  [{name}]')
    print(f'  Accuracy: {acc:.2f}% | Macro F1: {mf1:.4f} | Weighted F1: {wf1:.4f} | AUC: {mauc:.4f}')
    print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))
    return {'accuracy': acc, 'macro_f1': mf1, 'weighted_f1': wf1, 'macro_auc': mauc}


m_vit = print_metrics('ViT-Base (alone)', vit_preds, test_labels, vit_probs_test)
m_eff = print_metrics('EfficientNet-B3 (alone)', eff_preds, test_labels, eff_probs_test)
m_ens = print_metrics('Ensemble (weighted avg)', ens_preds_test, test_labels, ens_probs_test)

# Save results
results = {
    'ensemble_weight_vit': float(best_w),
    'ensemble_weight_eff': float(1 - best_w),
    'temperature': T_opt,
    'vit_metrics': m_vit,
    'efficientnet_metrics': m_eff,
    'ensemble_metrics': m_ens,
    'disagreement_rate': float(n_disagree / len(test_labels)),
    'training_time_minutes': train_time / 60,
}
with open(os.path.join(OUTPUT_DIR, 'ensemble_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f'\n  Results saved to {OUTPUT_DIR}/ensemble_results.json')
print(f'  EfficientNet checkpoint: {CHECKPOINT}')
print('=' * 65)
