# Phase 1 Agent Context — Script Details & Debugging Guide

## Common Model Loading Pattern (all scripts use this)
```python
import timm, torch, torch.nn as nn

class MultiTaskViT(nn.Module):
    def __init__(self, n_disease=5, n_severity=5, drop=0.3):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        self.drop = nn.Dropout(drop)
        self.disease_head = nn.Sequential(
            nn.Linear(768, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 5))
        self.severity_head = nn.Sequential(
            nn.Linear(768, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 5))
    def forward(self, x):
        f = self.backbone(x); f = self.drop(f)
        return self.disease_head(f), self.severity_head(f)

model = MultiTaskViT().to(DEVICE)
ckpt = torch.load('outputs_v3/best_model.pth', map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
```

## Common Data Loading Pattern
```python
# Test CSV columns: image_path, dataset, disease_label, severity_label, [cache_path, source]
# Note: some CSVs have 'dataset' column, others have 'source' — check both
test_df = pd.read_csv('data/test_split.csv')
# dataset column values: 'APTOS' or 'ODIR' (sometimes column is named 'source')

# Cache loading:
cache_fp = f'preprocessed_cache_v3/{stem}_{224}.npy'
img = np.load(cache_fp)  # (224, 224, 3) uint8

# Transform for inference:
normalize = transforms.Normalize([0.4298,0.2784,0.1559], [0.2857,0.2065,0.1465])
transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), normalize])
```

## Phase 1A: eval_dashboard.py
- **Script**: `/teamspace/studios/this_studio/eval_dashboard.py` (created by agent)
- **Purpose**: Confusion matrix, ROC curves, PR curves, calibration diagram, confidence histograms
- **Output dir**: `outputs_v3/evaluation/`
- **Expected outputs**: confusion_matrix.png, roc_curves_per_class.png, precision_recall_curves.png,
  calibration_reliability.png, confidence_histograms.png, error_analysis_by_source.png, metrics_report.json
- **Status**: Script created, was running on CPU (too slow), needs GPU re-run
- **Potential issues**:
  - Model returns tuple (d_out, s_out) — always unpack
  - Temperature scaling: logits / 0.6438 before softmax
  - Test CSV may have 'dataset' or 'source' column for APTOS/ODIR

## Phase 1B: mc_dropout_uncertainty.py
- **Script**: `/teamspace/studios/this_studio/mc_dropout_uncertainty.py` (created by agent)
- **Purpose**: MC Dropout with T=30 forward passes per image, uncertainty decomposition
- **Output dir**: `outputs_v3/uncertainty/`
- **Expected outputs**: uncertainty_vs_accuracy.png, rejection_curve.png,
  epistemic_vs_aleatoric.png, uncertainty_by_class.png, confidence_vs_uncertainty.png,
  mc_dropout_results.json
- **Status**: Script created, was running on CPU (too slow), needs GPU re-run
- **Key implementation**: Enable dropout layers with m.train() while keeping batchnorm in eval
- **Potential issues**:
  - Dropout rate is 0.3 (from model config) — should give meaningful variance
  - 30 passes x 1281 images = 38,430 forward passes total

## Phase 1C: integrated_gradients_xai.py
- **Script**: `/teamspace/studios/this_studio/integrated_gradients_xai.py` (created by agent)
- **Purpose**: Captum IntegratedGradients + comparison with Attention Rollout
- **Output dir**: `outputs_v3/xai/`
- **Expected outputs**: comparison_grid.png, ig_individual_*.png, agreement_heatmap.png,
  agreement_score.json
- **Status**: Script created, was running on CPU (too slow), needs GPU re-run
- **Key implementation**:
  - Need wrapper function for captum: input tensor → disease logits only (not tuple)
  - Baseline: blurred image (GaussianBlur sigma=10), not black (fundus has dark background)
  - ViTAttentionRollout class is in gradcam_v3.py (can import or copy)
- **Potential issues**:
  - Captum needs gradients enabled (no torch.no_grad)
  - Model wrapper must return single tensor, not tuple

## Phase 1D: fairness_analysis.py
- **Script**: `/teamspace/studios/this_studio/fairness_analysis.py` (created by agent)
- **Purpose**: Performance by dataset source (APTOS vs ODIR), domain gap analysis
- **Output dir**: `outputs_v3/fairness/`
- **Expected outputs**: performance_by_source.png, calibration_by_source.png,
  confusion_matrix_aptos.png, confusion_matrix_odir.png, confidence_by_source.png,
  error_patterns.png, domain_gap_report.json
- **Status**: Script created, was running on CPU (too slow), needs GPU re-run
- **Key note**: APTOS only has DR images (class 1). ODIR has all 5 classes.
  The interesting comparison is DR performance: APTOS-DR vs ODIR-DR.
- **Potential issues**:
  - CSV column name for source: check 'dataset' first, fallback to 'source'
  - APTOS images will only appear in class 1 (DR)

## Phase 2B: train_ensemble.py
- **Script**: `/teamspace/studios/this_studio/train_ensemble.py`
- **Purpose**: Train EfficientNet-B3, optimize ensemble weights with ViT
- **Output dir**: `outputs_v3/ensemble/`
- **Expected outputs**: efficientnet_b3.pth, ensemble_results.json
- **Status**: Script created, was training epoch 1 on CPU (very slow), needs GPU
- **Architecture**: EfficientNet-B3 (timm, 12M params), same training recipe as ViT
- **Ensemble**: Grid search w_vit in [0.1, 0.9], weighted average of probabilities

## Debugging Tips
1. If "CUDA out of memory": reduce batch_size to 16, or run scripts sequentially not in parallel
2. If model loading fails: check that MultiTaskViT class matches exactly (768-dim features)
3. If test CSV has no 'source' column: the column is named 'dataset' with values 'APTOS'/'ODIR'
4. If cache .npy files not found: they're in `preprocessed_cache_v3/` with pattern `{stem}_224.npy`
5. If norm stats error: keys are 'mean_rgb' and 'std_rgb' (not 'mean'/'std')
