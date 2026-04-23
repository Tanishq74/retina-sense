# RetinaSense-ViT — Test Specification (45 Test Cases)

**Project**: RetinaSense-ViT — Multi-disease retinal fundus classification
**Authors**: Tanishq Tamarkar, Rafae Mohammed Hussain, Dr. Revathi M — SRM Institute, Chennai
**Production Model**: DANN-v3 (`outputs_v3/dann_v3/best_model.pth`)
**Date**: 2026-03-27

---

## Coverage Map

| Category | Test IDs | Count |
|---|---|---|
| Preprocessing Pipeline | TC-01 – TC-05 | 5 |
| Model Architecture | TC-06 – TC-10 | 5 |
| Training Pipeline (DANN-v3) | TC-11 – TC-16 | 6 |
| Calibration & Threshold Optimization | TC-17 – TC-20 | 4 |
| FAISS Retrieval Index | TC-21 – TC-24 | 4 |
| RAD Evaluation Framework | TC-25 – TC-28 | 4 |
| Confidence Routing System | TC-29 – TC-32 | 4 |
| Explainability Modules | TC-33 – TC-35 | 3 |
| Uncertainty Quantification | TC-36 – TC-38 | 3 |
| Fairness & LODO Analysis | TC-39 – TC-41 | 3 |
| Gradio Demo & FastAPI Server | TC-42 – TC-45 | 4 |
| **Total** | | **45** |

---

## Section 1 — Preprocessing Pipeline

### TC-01: Ben Graham Enhancement — APTOS Source

| Field | Detail |
|---|---|
| **ID** | TC-01 |
| **Category** | Preprocessing |
| **Priority** | High |
| **Script** | `unified_preprocessing.py` → `ben_graham()` |

**Description**
Ben Graham enhancement is the correct preprocessing path for APTOS images. It removes vignetting and uneven illumination by subtracting a Gaussian-blurred low-frequency estimate from the original, then applying a circular mask.

**Input**
Any `.png` fundus image from the APTOS dataset. Source string = `"APTOS"`.

**Steps**
1. Call `preprocess_image(path, source='APTOS', sz=224)`.
2. Inspect the returned array.

**Expected Output**
- Shape: `(224, 224, 3)`, dtype `uint8`.
- The peripheral pixels outside the circular mask at radius `0.48 × 224 ≈ 107 px` are set to zero (black border).
- Center region exhibits amplified local contrast (formula: `4×img − 4×blurred + 128`).
- No negative pixel values; range `[0, 255]`.

**Pass Criteria**
- `output.shape == (224, 224, 3)`
- `output.dtype == np.uint8`
- Corner pixel `output[0, 0]` == `[0, 0, 0]` (masked region)
- Mean pixel value in the central 100×100 region > 60 (non-trivially processed)

**Fail Conditions**
- Shape mismatch, dtype float instead of uint8, no circular mask applied, or identical to raw resize-only output.

---

### TC-02: CLAHE Preprocessing — ODIR / MESSIDOR-2 Sources

| Field | Detail |
|---|---|
| **ID** | TC-02 |
| **Category** | Preprocessing |
| **Priority** | High |
| **Script** | `unified_preprocessing.py` → `clahe_preprocess()` |

**Description**
ODIR and MESSIDOR-2 images use CLAHE on the LAB L-channel only. This normalizes local contrast without altering hue or globally oversaturating colors.

**Input**
ODIR image with source `"ODIR"`. Same image with source `"MESSIDOR-2"`.

**Steps**
1. Call `preprocess_image(path, source='ODIR', sz=224)`.
2. Call `preprocess_image(path, source='MESSIDOR-2', sz=224)`.
3. Compare to raw resized image.

**Expected Output**
- Both calls return shape `(224, 224, 3)`, dtype `uint8`.
- Circular mask applied (corners = 0).
- Standard deviation of the L channel (LAB) is higher than raw image L channel (CLAHE expands contrast).
- Hue (H channel in HSV) distribution is nearly identical to raw image (only luminance was modified).

**Pass Criteria**
- `output.shape == (224, 224, 3)` for both sources
- `np.std(clahe_L) > np.std(raw_L)` (contrast enhanced)
- `|mean_H_clahe − mean_H_raw| < 5` (hue unchanged, in degrees)

---

### TC-03: Resize-Only Preprocessing — REFUGE2 Source

| Field | Detail |
|---|---|
| **ID** | TC-03 |
| **Category** | Preprocessing |
| **Priority** | Medium |
| **Script** | `unified_preprocessing.py` → `resize_only()` |

**Description**
REFUGE2 images from the Zeiss Visucam 500 are already standardized quality. Only resize and circular masking are applied.

**Input**
REFUGE2 image with source `"REFUGE2"`.

**Steps**
1. Call `preprocess_image(path, source='REFUGE2', sz=224)`.

**Expected Output**
- Shape `(224, 224, 3)`, dtype `uint8`.
- Interior pixel values closely match a bilinear-resized version of the original (no contrast modification).
- Circular mask applied at radius ≈ 107 px.

**Pass Criteria**
- `output.shape == (224, 224, 3)`
- Central 100×100 SSIM between output and plain-resized ≥ 0.97 (minimal modification beyond masking)
- Corner pixel = `[0, 0, 0]`

---

### TC-04: Cache Load and Fallback Chain

| Field | Detail |
|---|---|
| **ID** | TC-04 |
| **Category** | Preprocessing |
| **Priority** | High |
| **Script** | `unified_preprocessing.py` / Dataset loader |

**Description**
The cache loader attempts three locations before falling back to on-the-fly preprocessing. Cache key format is `{stem}_{img_size}.npy`.

**Input**
An image stem `"sample_001"` that exists at `preprocessed_cache_unified/sample_001_224.npy`.

**Steps**
1. Ensure `preprocessed_cache_unified/sample_001_224.npy` exists.
2. Call the dataset `__getitem__` for this sample.
3. Verify it loads from cache (not recomputed).
4. Delete the cache file.
5. Call `__getitem__` again — should fall back to on-the-fly preprocessing.

**Expected Output**
- First call: loads `.npy` file, returns identical array (byte-for-byte match).
- Fallback call: returns same shape `(224, 224, 3)` array via preprocessing.
- No `FileNotFoundError` raised; fallback is silent.

**Pass Criteria**
- Cache hit: loaded array `== np.load(cache_path)`
- Fallback: no exception raised, `shape == (224, 224, 3)`

---

### TC-05: Normalization Statistics Loading with Fallback

| Field | Detail |
|---|---|
| **ID** | TC-05 |
| **Category** | Preprocessing |
| **Priority** | Medium |
| **Script** | `unified_preprocessing.py`, `app.py` |

**Description**
Fundus-specific normalization stats are loaded from `configs/fundus_norm_stats_unified.json`. If missing, the system falls back to ImageNet defaults `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`.

**Input**
Normal execution (file present). Then rename the file to simulate absence.

**Steps**
1. Load stats normally — verify fundus-specific values are used.
2. Temporarily rename `configs/fundus_norm_stats_unified.json`.
3. Re-initialize the loader — verify ImageNet defaults are returned.
4. Restore the file.

**Expected Output**
- Normal: `mean_rgb` and `std_rgb` keys present, values differ from ImageNet defaults.
- Fallback: mean = `[0.485, 0.456, 0.406]`, std = `[0.229, 0.224, 0.225]`.

**Pass Criteria**
- Primary path returns non-ImageNet values
- Fallback path returns exact ImageNet defaults
- No crash in either path

---

## Section 2 — Model Architecture

### TC-06: ViT-Base Backbone Output Shape

| Field | Detail |
|---|---|
| **ID** | TC-06 |
| **Category** | Model Architecture |
| **Priority** | High |
| **Script** | `app.py` → `MultiTaskViT` |

**Description**
The `timm` ViT-Base/16 backbone with `num_classes=0` returns the 768-dimensional CLS token embedding for a `(B, 3, 224, 224)` input batch.

**Input**
Random tensor `x = torch.randn(4, 3, 224, 224)` passed to `model.backbone(x)`.

**Steps**
1. Instantiate `MultiTaskViT()`.
2. Call `features = model.backbone(x)`.
3. Check shape.

**Expected Output**
- `features.shape == (4, 768)` — batch of 4, CLS embedding dim 768.

**Pass Criteria**
- `features.shape == torch.Size([4, 768])`
- No runtime error on CPU or CUDA

---

### TC-07: Disease Head Output Shape and Logit Range

| Field | Detail |
|---|---|
| **ID** | TC-07 |
| **Category** | Model Architecture |
| **Priority** | High |
| **Script** | `app.py` → `MultiTaskViT.disease_head` |

**Description**
The disease head maps 768-dim features to 5-class logits through `768 → 512 → 256 → 5` with BatchNorm, ReLU, and Dropout layers.

**Input**
Random feature tensor `f = torch.randn(8, 768)`.

**Steps**
1. Set `model.eval()`.
2. Call `logits = model.disease_head(f)`.
3. Check shape and range.

**Expected Output**
- `logits.shape == (8, 5)`
- Values are unbounded logits (can be negative or large positive — no softmax yet)
- After `softmax`, each row sums to 1.0

**Pass Criteria**
- `logits.shape == torch.Size([8, 5])`
- `torch.allclose(F.softmax(logits, dim=1).sum(dim=1), torch.ones(8), atol=1e-5)`

---

### TC-08: Severity Head Output Shape

| Field | Detail |
|---|---|
| **ID** | TC-08 |
| **Category** | Model Architecture |
| **Priority** | Medium |
| **Script** | `app.py` → `MultiTaskViT.severity_head` |

**Description**
The severity head maps 768-dim features to 5 DR severity grade logits (`768 → 256 → 5`). It is supervised only for APTOS samples (others have `severity_label = -1`).

**Input**
Random feature tensor `f = torch.randn(4, 768)`.

**Steps**
1. Call `severity_logits = model.severity_head(f)`.
2. Check shape.

**Expected Output**
- `severity_logits.shape == (4, 5)` — 5 DR severity levels (0–4).

**Pass Criteria**
- `severity_logits.shape == torch.Size([4, 5])`

---

### TC-09: Gradient Reversal Layer — Forward and Backward

| Field | Detail |
|---|---|
| **ID** | TC-09 |
| **Category** | Model Architecture |
| **Priority** | High |
| **Script** | `train_dann_v3.py` → `GradientReversalFunction` |

**Description**
The GRL passes the input unchanged during the forward pass. During backprop, it negates gradients scaled by `alpha`. This forces domain-invariant feature learning.

**Input**
Tensor `x = torch.randn(4, 768, requires_grad=True)`, `alpha = 0.3`.

**Steps**
1. Compute `y = GradientReversalFunction.apply(x, 0.3)`.
2. Verify `y` equals `x` (forward pass identity).
3. Compute `loss = y.sum()` and call `loss.backward()`.
4. Verify `x.grad == -0.3 * torch.ones_like(x)`.

**Expected Output**
- Forward: `y` is numerically identical to `x`
- Backward: `x.grad == -alpha` everywhere (negated and scaled)

**Pass Criteria**
- `torch.allclose(y, x)`
- `torch.allclose(x.grad, -0.3 * torch.ones_like(x))`

---

### TC-10: DANN-v3 Checkpoint Loading — Key Filtering

| Field | Detail |
|---|---|
| **ID** | TC-10 |
| **Category** | Model Architecture |
| **Priority** | High |
| **Script** | `app.py` lines 89–100 |

**Description**
When loading a DANN-v3 checkpoint into `MultiTaskViT` (which has no domain head), keys prefixed with `domain_head.*` and `grl.*` must be filtered out. The remaining keys must load without missing or unexpected keys.

**Input**
`outputs_v3/dann_v3/best_model.pth` checkpoint.

**Steps**
1. Load checkpoint: `ckpt = torch.load(...)`.
2. Filter: remove keys starting with `'domain_head'` or `'grl'`.
3. Call `model.load_state_dict(filtered, strict=False)`.
4. Check `load_result.missing_keys` and `load_result.unexpected_keys`.

**Expected Output**
- `len(load_result.missing_keys) == 0`
- `len(load_result.unexpected_keys) == 0`
- Filtered key count > 0 (some DANN keys were removed)

**Pass Criteria**
- No missing keys
- No unexpected keys
- Model inference runs without error post-load

---

## Section 3 — Training Pipeline (DANN-v3)

### TC-11: Focal Loss with Label Smoothing — Numerical Correctness

| Field | Detail |
|---|---|
| **ID** | TC-11 |
| **Category** | Training Pipeline |
| **Priority** | High |
| **Script** | `train_dann_v3.py` → `FocalLoss` |

**Description**
The Focal Loss applies `(1 − pt)^gamma × CE` with per-class alpha weights and label smoothing = 0.1. For a perfectly confident correct prediction (`pt → 1`), focal weight `→ 0`.

**Input**
- `logits = torch.tensor([[10.0, -1.0, -1.0, -1.0, -1.0]])` (very confident class 0)
- `targets = torch.tensor([0])`
- `gamma = 2.0`, `alpha = [1.0, 1.0, 1.0, 1.0, 1.0]`, `label_smoothing = 0.1`

**Expected Output**
- Loss is near zero (but not exactly 0 due to label smoothing floor)
- `0 < loss < 0.05`

**Pass Criteria**
- `loss.item() < 0.05`
- `loss.item() >= 0`

---

### TC-12: Ganin Lambda Schedule — Boundary Values

| Field | Detail |
|---|---|
| **ID** | TC-12 |
| **Category** | Training Pipeline |
| **Priority** | Medium |
| **Script** | `train_dann_v3.py` → `ganin_lambda()` |

**Description**
The Ganin lambda ramps from 0 at epoch 0 to a cap of `max_lambda=0.3`. The raw sigmoid would exceed 0.3, so the `min(raw, max_lambda)` cap ensures stability.

**Input**
`total_epochs = 40`, `max_lambda = 0.3`

**Steps**
1. `ganin_lambda(0, 40, 0.3)` — at start
2. `ganin_lambda(20, 40, 0.3)` — midpoint
3. `ganin_lambda(39, 40, 0.3)` — near end

**Expected Output**
- Epoch 0: `≈ 0.0` (near zero)
- Epoch 20: between 0.1 and 0.3
- Epoch 39: exactly `0.3` (capped)

**Pass Criteria**
- `ganin_lambda(0, 40) < 0.05`
- `0.1 < ganin_lambda(20, 40) < 0.3`
- `ganin_lambda(39, 40) == 0.3`

---

### TC-13: Progressive DR Alpha Schedule — Monotonic Increase

| Field | Detail |
|---|---|
| **ID** | TC-13 |
| **Category** | Training Pipeline |
| **Priority** | Medium |
| **Script** | `train_dann_v3.py` — DR alpha schedule |

**Description**
The DR alpha boost ramps from `1.5×` at epoch 0 to `3.0×` at the final epoch. The alpha for all other classes remains at their base values. At epoch 0, DR alpha = `base_alpha[1] × 1.5`. At final epoch, DR alpha = `base_alpha[1] × 3.0`.

**Input**
`alpha_start=1.5`, `alpha_end=3.0`, `total_epochs=40`, base DR alpha = 1.0

**Steps**
Compute DR alpha at epoch 0, 20, 39.

**Expected Output**
- Epoch 0: DR alpha = 1.5
- Epoch 20: DR alpha ≈ 2.25
- Epoch 39: DR alpha = 3.0
- Non-DR class alphas remain constant throughout

**Pass Criteria**
- Monotonically non-decreasing
- Exact boundary values: 1.5 at epoch 0, 3.0 at epoch `N-1`

---

### TC-14: MixUp Augmentation — Label Mixing Correctness

| Field | Detail |
|---|---|
| **ID** | TC-14 |
| **Category** | Training Pipeline |
| **Priority** | High |
| **Script** | `train_dann_v3.py` — MixUp block |

**Description**
MixUp mixes image pixel values with `lam` from `Beta(0.2, 0.2)`. The disease loss is a weighted combination of the two label losses. Domain labels are NOT mixed — domain discriminator always sees original features.

**Input**
Batch of 4 images `x`, disease labels `y = [0, 1, 2, 3]`, `lam = 0.7`.

**Steps**
1. Apply MixUp: `mixed_x = 0.7 * x + 0.3 * x[perm]`.
2. Compute mixed loss: `L = 0.7 * CE(logits, y_a) + 0.3 * CE(logits, y_b)`.

**Expected Output**
- `mixed_x.shape == x.shape` (no shape change)
- Pixel values in `[0, 1]` range (if input was normalized)
- Loss is a valid scalar > 0

**Pass Criteria**
- `mixed_x.shape == x.shape`
- `0 < mixed_loss_scalar < 10`
- Domain labels `d` remain identical to original batch (not mixed)

---

### TC-15: Checkpoint Save and Resume — Key Fields Present

| Field | Detail |
|---|---|
| **ID** | TC-15 |
| **Category** | Training Pipeline |
| **Priority** | High |
| **Script** | `train_dann_v3.py` — checkpoint saving |

**Description**
The checkpoint saved when macro-F1 improves contains mandatory fields: `epoch`, `model_state_dict`, `val_acc`, `macro_f1`, `domain_map`, `num_domains`, `history`, `args`, `dr_alpha_boost`.

**Input**
`outputs_v3/dann_v3/best_model.pth`

**Steps**
1. `ckpt = torch.load('outputs_v3/dann_v3/best_model.pth', map_location='cpu')`.
2. Inspect all required keys.

**Expected Output**
All 9 keys present. `num_domains == 4`. `macro_f1` matches reported value `≈ 0.879–0.886`. `model_state_dict` contains ViT backbone weights.

**Pass Criteria**
- All 9 required keys present in checkpoint dict
- `ckpt['num_domains'] == 4`
- `ckpt['macro_f1'] >= 0.87`
- `'backbone.cls_token' in ckpt['model_state_dict']`

---

### TC-16: 8-Way TTA — Augmentation Coverage

| Field | Detail |
|---|---|
| **ID** | TC-16 |
| **Category** | Training Pipeline |
| **Priority** | Medium |
| **Script** | `train_dann_v3.py` / `tta_evaluation.py` — TTA block |

**Description**
TTA applies 8 transforms: identity, H-flip, V-flip, H+V flip, 90° rotation, 180° rotation, 270° rotation, center crop→224. Probabilities from all 8 passes are averaged before applying temperature scaling.

**Input**
A single test image tensor `(1, 3, 224, 224)`.

**Steps**
1. Apply all 8 TTA transforms.
2. Pass each through the model.
3. Average the 8 probability vectors.

**Expected Output**
- 8 distinct tensors (transforms must differ from each other)
- Averaged probability vector sums to 1.0
- Shape `(1, 5)` after averaging

**Pass Criteria**
- Number of unique transforms = 8 (at least transforms 2–7 differ from identity)
- `avg_probs.sum() ≈ 1.0` (within 1e-5)
- `avg_probs.shape == (1, 5)`

---

## Section 4 — Calibration & Threshold Optimization

### TC-17: Temperature Scaling — Calibration Effect

| Field | Detail |
|---|---|
| **ID** | TC-17 |
| **Category** | Calibration |
| **Priority** | High |
| **Script** | `configs/temperature.json` value application |

**Description**
The optimal temperature `T = 0.5657` is stored in `configs/temperature.json`. Dividing logits by T before softmax sharpens the distribution (T < 1). The ECE drops from `0.149` pre-scaling to `0.037` post-scaling.

**Input**
Raw logits `[3.0, 1.0, 0.5, 0.3, 0.2]`, T = 0.5657.

**Steps**
1. `probs_raw = softmax(logits)`
2. `probs_scaled = softmax(logits / T)`
3. Compare max probability.

**Expected Output**
- `max(probs_scaled) > max(probs_raw)` (distribution is sharpened)
- Top class is the same in both (rank order preserved)
- `probs_scaled.sum() ≈ 1.0`

**Pass Criteria**
- `probs_scaled.argmax() == probs_raw.argmax()`
- `probs_scaled.max() > probs_raw.max()`
- Temperature value loaded from `configs/temperature.json` is `≈ 0.5657`

---

### TC-18: Per-Class Threshold Grid Search — Correctness

| Field | Detail |
|---|---|
| **ID** | TC-18 |
| **Category** | Calibration |
| **Priority** | High |
| **Script** | Threshold optimization logic |

**Description**
For each class, a grid of 50 thresholds in `[0.05, 0.95]` is searched independently to maximize binary F1 for that class. The production thresholds from `configs/thresholds.json` are:
- Normal: 0.3990, DR: 0.4173, Glaucoma: 0.6745, Cataract: 0.2888, AMD: 0.3439.

**Input**
Load `configs/thresholds.json`.

**Steps**
1. Parse the file.
2. Verify 5 threshold values are in `[0.05, 0.95]`.
3. Verify class names match expected order.
4. Note: Glaucoma threshold (0.6745) is the highest, reflecting lower base probability for the smallest class.

**Expected Output**
- 5 threshold values all in `(0.0, 1.0)`
- Glaucoma threshold > all other thresholds (highest, as Glaucoma is the hardest class with lowest natural probability)
- Class names: `["Normal", "Diabetes/DR", "Glaucoma", "Cataract", "AMD"]`

**Pass Criteria**
- All 5 values in `(0.05, 0.95)`
- `thresholds[2] > thresholds[0]` (Glaucoma > Normal)
- `thresholds[2] > thresholds[1]` (Glaucoma > DR)
- Class names match expected list exactly

---

### TC-19: Threshold Application — Fallback to Argmax

| Field | Detail |
|---|---|
| **ID** | TC-19 |
| **Category** | Calibration |
| **Priority** | Medium |
| **Script** | `apply_thresholds()` function |

**Description**
`apply_thresholds()` selects the class with the highest probability among those exceeding their per-class threshold. If no class exceeds its threshold (e.g., very low-confidence image), it falls back to `argmax(probs)`.

**Input**
- Case A: `probs = [0.5, 0.5, 0.8, 0.4, 0.5]`, thresholds = `[0.399, 0.417, 0.674, 0.289, 0.344]`
  Classes above threshold: index 2 (0.8 > 0.674). Predicted: 2 (Glaucoma).
- Case B: `probs = [0.1, 0.15, 0.2, 0.1, 0.1]`, same thresholds
  No class exceeds threshold. Falls back to argmax: index 2 (0.2).

**Expected Output**
- Case A: predicted class = 2 (Glaucoma)
- Case B: predicted class = 2 (argmax fallback)

**Pass Criteria**
- Case A returns 2
- Case B returns 2 via argmax fallback with no exception raised

---

### TC-20: ECE Computation — Bin Counting Accuracy

| Field | Detail |
|---|---|
| **ID** | TC-20 |
| **Category** | Calibration |
| **Priority** | Medium |
| **Script** | ECE computation in calibration logic |

**Description**
ECE uses 15 equal-width bins. For a perfectly calibrated model (confidence = accuracy in every bin), ECE = 0. For a model whose confidence is always exactly equal to accuracy, the weighted sum of `|accuracy − confidence|` per bin must be 0.

**Input**
Synthetic: `probs` where confidence (max prob) = accuracy per bin.

**Steps**
1. Create 100 samples where `confidence = 0.7` and 70 are correct.
2. Compute ECE.

**Expected Output**
- ECE ≈ 0.0 (perfectly calibrated synthetic data)
- Production ECE from `configs/temperature.json` = `ece_after ≈ 0.037`

**Pass Criteria**
- Synthetic test: `ECE < 0.01`
- Loaded `ece_after` value `< 0.05`
- `ece_after < ece_before` (calibration improved ECE)

---

## Section 5 — FAISS Retrieval Index

### TC-21: Index File Integrity

| Field | Detail |
|---|---|
| **ID** | TC-21 |
| **Category** | FAISS Retrieval |
| **Priority** | High |
| **Script** | `rebuild_faiss_full.py` output |

**Description**
The FAISS index at `outputs_v3/retrieval/index_flat_ip.faiss` must contain 8,241 vectors of dimension 768, using `IndexFlatIP` (inner-product / cosine after L2-normalization).

**Input**
`outputs_v3/retrieval/index_flat_ip.faiss`

**Steps**
1. `index = faiss.read_index('outputs_v3/retrieval/index_flat_ip.faiss')`.
2. Check `index.ntotal` and `index.d`.

**Expected Output**
- `index.ntotal == 8241`
- `index.d == 768`
- Index type is `IndexFlatIP`

**Pass Criteria**
- `index.ntotal == 8241`
- `index.d == 768`
- `isinstance(index, faiss.IndexFlatIP)` or cast succeeds

---

### TC-22: Metadata JSON — All 5 Classes Present

| Field | Detail |
|---|---|
| **ID** | TC-22 |
| **Category** | FAISS Retrieval |
| **Priority** | High |
| **Script** | `outputs_v3/retrieval/metadata.json` |

**Description**
After the full rebuild (`rebuild_faiss_full.py`), all 5 disease classes (Normal, DR, Glaucoma, Cataract, AMD) must be represented in the 8,241-entry metadata file. A prior bug caused AMD to be missing entirely, which broke AMD retrieval.

**Input**
`outputs_v3/retrieval/metadata.json`

**Steps**
1. Load the JSON list of 8,241 dicts.
2. Collect all unique `label` values.
3. Verify all 5 class indices `{0, 1, 2, 3, 4}` are present.

**Expected Output**
- `len(metadata) == 8241`
- Set of unique labels = `{0, 1, 2, 3, 4}`
- Each dict has keys: `label`, `class_name`, `source`, `image_path`, `cache_path`

**Pass Criteria**
- `len(metadata) == 8241`
- `set(m['label'] for m in metadata) == {0, 1, 2, 3, 4}`
- All 5 class names appear: Normal, Diabetes/DR, Glaucoma, Cataract, AMD

---

### TC-23: L2 Normalization Before Search

| Field | Detail |
|---|---|
| **ID** | TC-23 |
| **Category** | FAISS Retrieval |
| **Priority** | High |
| **Script** | Retrieval inference logic |

**Description**
Because the index uses `IndexFlatIP`, correct cosine similarity requires all stored and query vectors to be L2-normalized first. Inner product of two unit vectors equals cosine similarity. Un-normalized vectors would give biased distances.

**Input**
A 768-dim query embedding `q` extracted from the model backbone.

**Steps**
1. Extract `q = model.backbone(img_tensor)` — shape `(1, 768)`.
2. Compute L2 norm: `norm = np.linalg.norm(q)`.
3. Apply: `faiss.normalize_L2(q)`.
4. Verify `np.linalg.norm(q) ≈ 1.0`.
5. Search index: `D, I = index.search(q, k=5)`.
6. Verify all distances `D[0]` are in `[−1, 1]`.

**Expected Output**
- Post-normalization: `‖q‖ ≈ 1.0` (within 1e-6)
- All 5 returned distances in `[−1, 1]` (valid cosine similarities)
- `I.shape == (1, 5)`

**Pass Criteria**
- `abs(np.linalg.norm(q) - 1.0) < 1e-5`
- `all(-1.0 <= d <= 1.0 for d in D[0])`
- `I.shape == (1, 5)`

---

### TC-24: Top-5 Retrieval — Same-Class Hit Rate

| Field | Detail |
|---|---|
| **ID** | TC-24 |
| **Category** | FAISS Retrieval |
| **Priority** | High |
| **Script** | `rad_evaluation.py` |

**Description**
Recall@1 on the test set is 94.0%: for 94% of test queries, the nearest neighbor belongs to the same class. Recall@5 is 98.5%. This test validates the retrieval quality on a small held-out sample.

**Input**
10 test images from each class (50 total), queried against the full index.

**Steps**
1. Extract 50 embeddings.
2. For each, retrieve top-5 nearest neighbors.
3. Check if label of nearest neighbor (rank 1) matches query label.

**Expected Output**
- At least 40/50 queries have a matching class at rank 1 (≥ 80% recall@1 on mini-sample, given the production value is 94%)
- All 5 classes have at least one correct retrieval

**Pass Criteria**
- Recall@1 ≥ 0.80 on this 50-sample mini-test
- No class returns 0 correct retrievals

---

## Section 6 — RAD Evaluation Framework

### TC-25: MAP Computation — Formula Verification

| Field | Detail |
|---|---|
| **ID** | TC-25 |
| **Category** | RAD Evaluation |
| **Priority** | High |
| **Script** | `rad_evaluation.py` → MAP computation |

**Description**
Mean Average Precision (MAP) = mean of per-query Average Precision. For a query where the first 3 of 5 retrieved are correct (positions 1, 2, 3), AP = (1/1 + 2/2 + 3/3) / 3 = 1.0. Production MAP = 0.921.

**Input**
Synthetic: 2 queries.
- Query 1: retrieved labels `[0, 0, 1, 0, 2]`, true label `0` → AP = (1/1 + 2/2 + 3/4) / 3 = (1 + 1 + 0.75)/3 = 0.917.
- Query 2: retrieved labels `[1, 1, 1, 1, 1]`, true label `1` → AP = 1.0.

**Steps**
Compute MAP manually and programmatically, compare.

**Expected Output**
- Query 1 AP ≈ 0.917
- Query 2 AP = 1.0
- MAP = (0.917 + 1.0) / 2 = 0.958

**Pass Criteria**
- Computed MAP matches expected to within 0.001
- Production MAP from `outputs_v3/retrieval/rad_evaluation_results.json` ≥ 0.90

---

### TC-26: RAD Combined Accuracy > Standalone

| Field | Detail |
|---|---|
| **ID** | TC-26 |
| **Category** | RAD Evaluation |
| **Priority** | High |
| **Script** | `rad_evaluation.py` → RAD combined predictions |

**Description**
RAD combines model softmax probabilities (`alpha=0.5`) with kNN similarity-weighted votes. This achieves 94.0% accuracy (+4.9% over standalone 89.09%). The combination formula: `combined = 0.5 × model_probs + 0.5 × knn_probs`.

**Input**
`outputs_v3/retrieval/rad_evaluation_results.json`

**Steps**
1. Load JSON results.
2. Compare `rad_accuracy` vs `standalone_accuracy`.

**Expected Output**
- `rad_accuracy ≥ 0.93`
- `rad_accuracy > standalone_accuracy`
- Improvement ≥ 4%

**Pass Criteria**
- `rad_accuracy >= 0.93`
- `rad_accuracy - standalone_accuracy >= 0.04`

---

### TC-27: Agreement Analysis — High Agreement → High Accuracy

| Field | Detail |
|---|---|
| **ID** | TC-27 |
| **Category** | RAD Evaluation |
| **Priority** | Medium |
| **Script** | `rad_evaluation.py` → agreement analysis |

**Description**
When model prediction and kNN majority vote agree, accuracy is 97.3%. When they disagree, accuracy drops to 61.2%. This 36.1% gap validates that retrieval disagreement is a reliable uncertainty signal.

**Input**
`outputs_v3/retrieval/rad_evaluation_results.json`

**Steps**
1. Load `agreement_accuracy` and `disagreement_accuracy`.

**Expected Output**
- `agreement_accuracy ≥ 0.95`
- `disagreement_accuracy ≤ 0.70`
- `agreement_accuracy − disagreement_accuracy ≥ 0.25`

**Pass Criteria**
- All three inequalities satisfied

---

### TC-28: Per-Class AP — DR Highest, Glaucoma Lowest

| Field | Detail |
|---|---|
| **ID** | TC-28 |
| **Category** | RAD Evaluation |
| **Priority** | Medium |
| **Script** | `rad_evaluation.py` results |

**Description**
Per-class Average Precision from production: DR 0.952, Normal 0.906, Cataract 0.833, AMD 0.819, Glaucoma 0.742. DR is highest due to its distinctive lesion patterns. Glaucoma is lowest because optic disc cupping can resemble Normal variation.

**Input**
`outputs_v3/retrieval/rad_evaluation_results.json`

**Steps**
1. Load per-class AP values.

**Expected Output**
- DR AP > Normal AP > Cataract AP
- Glaucoma AP is the lowest of all 5 classes
- All AP values in `(0.5, 1.0)`

**Pass Criteria**
- `ap['DR'] > ap['Normal']`
- `ap['Glaucoma'] < ap['AMD']`
- `ap['Glaucoma'] == min(all AP values)`

---

## Section 7 — Confidence Routing System

### TC-29: ESCALATE Routing — Low Confidence Trigger

| Field | Detail |
|---|---|
| **ID** | TC-29 |
| **Category** | Confidence Routing |
| **Priority** | High |
| **Script** | `confidence_routing.py` → `ConfidenceRouter.route()` |

**Description**
Any prediction with `confidence < conf_low = 0.50` is escalated to specialist review regardless of entropy or retrieval agreement. This is the safety net for genuinely uncertain predictions.

**Input**
`confidence = 0.45`, `entropy = 0.3` (low), `retrieval_agrees = True`

**Steps**
Call `router.route(confidence=0.45, entropy=0.3, retrieval_agrees=True)`.

**Expected Output**
- Returns `"ESCALATE"` (confidence below threshold overrides everything else)

**Pass Criteria**
- Return value == `"ESCALATE"`

---

### TC-30: AUTO-REPORT Routing — All Conditions Met

| Field | Detail |
|---|---|
| **ID** | TC-30 |
| **Category** | Confidence Routing |
| **Priority** | High |
| **Script** | `confidence_routing.py` → `ConfidenceRouter.route()` |

**Description**
AUTO-REPORT requires ALL three conditions: `confidence ≥ 0.85`, `entropy < 0.5 nats`, AND `retrieval_agrees = True`. The production AUTO-REPORT tier covers 76.9% of cases at 96.8% accuracy.

**Input**
`confidence = 0.92`, `entropy = 0.2`, `retrieval_agrees = True`

**Steps**
Call `router.route(confidence=0.92, entropy=0.2, retrieval_agrees=True)`.

**Expected Output**
- Returns `"AUTO-REPORT"`

**Pass Criteria**
- Return value == `"AUTO-REPORT"`

**Boundary test**: Set `retrieval_agrees = False` with same confidence/entropy → expect `"REVIEW"` (not AUTO-REPORT).

---

### TC-31: REVIEW Routing — Mid-Range Confidence

| Field | Detail |
|---|---|
| **ID** | TC-31 |
| **Category** | Confidence Routing |
| **Priority** | High |
| **Script** | `confidence_routing.py` → `ConfidenceRouter.route()` |

**Description**
Cases that are not ESCALATE and not AUTO-REPORT fall to REVIEW. This includes confident predictions where retrieval disagrees, or predictions with moderate confidence.

**Input**
- Case A: `confidence = 0.90`, `entropy = 0.2`, `retrieval_agrees = False` → REVIEW (retrieval disagrees)
- Case B: `confidence = 0.70`, `entropy = 0.4`, `retrieval_agrees = True` → REVIEW (confidence below AUTO-REPORT threshold)

**Expected Output**
- Both cases return `"REVIEW"`

**Pass Criteria**
- Case A: `"REVIEW"`
- Case B: `"REVIEW"`

---

### TC-32: Routing Distribution — Production Fractions

| Field | Detail |
|---|---|
| **ID** | TC-32 |
| **Category** | Confidence Routing |
| **Priority** | Medium |
| **Script** | `outputs_v3/retrieval/confidence_routing_results.json` |

**Description**
Production routing on the test set routes 76.9% AUTO-REPORT, 21.4% REVIEW, 1.7% ESCALATE. AUTO-REPORT accuracy = 96.8%, error catch rate = 77.2%.

**Input**
`outputs_v3/retrieval/confidence_routing_results.json`

**Steps**
1. Load file.
2. Verify tier fractions.
3. Verify accuracy values.

**Expected Output**
- AUTO-REPORT fraction ≥ 0.70
- ESCALATE fraction ≤ 0.05
- AUTO-REPORT accuracy ≥ 0.95
- Error catch rate ≥ 0.70 (fraction of errors routed to REVIEW/ESCALATE)

**Pass Criteria**
- `auto_report_fraction >= 0.70`
- `escalate_fraction <= 0.05`
- `auto_report_accuracy >= 0.95`
- `error_catch_rate >= 0.70`

---

## Section 8 — Explainability Modules

### TC-33: Attention Map Shape and Overlay

| Field | Detail |
|---|---|
| **ID** | TC-33 |
| **Category** | Explainability |
| **Priority** | High |
| **Script** | `app.py` → `get_attention_map()` |

**Description**
The `get_attention_map()` function extracts attention from the last ViT transformer block, averages over 12 heads, takes the CLS→patch attention slice (196 tokens), reshapes to `14×14`, and bilinear-upsamples to `224×224`.

**Input**
A `(1, 3, 224, 224)` image tensor passed through the model.

**Steps**
1. Register a forward hook on `model.backbone.blocks[-1].attn`.
2. Forward pass the image.
3. Extract `attn[0, 0, 1:]` (CLS row, all 196 patches).
4. Reshape to `(14, 14)`.
5. Resize to `(224, 224)`.

**Expected Output**
- Returned attention map: shape `(224, 224)`, dtype `float32`
- Values in `[0, 1]` (or non-negative — unnormalized attention weights)
- Non-uniform: not all pixels the same value (model attends to specific regions)

**Pass Criteria**
- `attn_map.shape == (224, 224)`
- `attn_map.min() >= 0`
- `attn_map.std() > 0.001` (non-trivial spatial variation)

---

### TC-34: Integrated Gradients — Completeness Axiom

| Field | Detail |
|---|---|
| **ID** | TC-34 |
| **Category** | Explainability |
| **Priority** | Medium |
| **Script** | `integrated_gradients_xai.py` |

**Description**
Integrated Gradients satisfies the completeness axiom: the sum of all pixel attributions equals the difference in model output between the input and baseline. `sum(IG) ≈ F(input) − F(baseline)`.

**Input**
A test image `x`, black baseline `x0 = zeros_like(x)`, target class `c`, 50 interpolation steps.

**Steps**
1. Compute IG attributions for class `c`.
2. Compute `F(x)[c] − F(x0)[c]` where `F` is the model logit/probability for class `c`.
3. Compare `sum(IG)` to the difference.

**Expected Output**
- `|sum(IG) − (F(x)[c] − F(x0)[c])| < 0.01` (completeness within numerical tolerance)

**Pass Criteria**
- Absolute difference between sum of attributions and output difference < 1% of output range

---

### TC-35: Grad-CAM Attention Rollout — 14×14 Patch Grid

| Field | Detail |
|---|---|
| **ID** | TC-35 |
| **Category** | Explainability |
| **Priority** | Medium |
| **Script** | `gradcam_v3.py` |

**Description**
Attention rollout for ViT-Base/16 produces a `14×14` spatial attention map (196 patches = 14×14 grid). Recursively multiplying 12-block attention matrices ensures the rollout captures indirect attention flows through all layers.

**Input**
A test image passed through `gradcam_v3.py`.

**Steps**
1. Run Grad-CAM / attention rollout.
2. Check intermediate map shape before final upsampling.

**Expected Output**
- Intermediate map: `(14, 14)`
- Final output (after `cv2.resize`): `(224, 224)`
- Visualization: `.png` file written to `outputs_v3/gradcam/`

**Pass Criteria**
- Intermediate shape `(14, 14)`
- Final shape `(224, 224)`
- Output file exists and is a valid PNG

---

## Section 9 — Uncertainty Quantification

### TC-36: MC Dropout — Backbone Deterministic, Head Stochastic

| Field | Detail |
|---|---|
| **ID** | TC-36 |
| **Category** | Uncertainty |
| **Priority** | High |
| **Script** | `confidence_routing.py` → MC Dropout block (section 10.3) |

**Description**
MC Dropout runs the backbone once (deterministic, frozen), then the disease head 15 times with dropout active. This correctly attributes uncertainty to the classifier head without re-running the expensive transformer 15 times.

**Input**
Any test image tensor `(1, 3, 224, 224)`.

**Steps**
1. Set model to `eval()`.
2. `features = model.backbone(img_tensor)` — repeat 5 times, verify identical output (deterministic).
3. Enable dropout: set all `nn.Dropout` modules to `train()`.
4. Run `model.disease_head(model.drop(features))` 15 times — verify outputs differ (stochastic).

**Expected Output**
- Backbone: all 5 feature tensors identical (`max diff < 1e-7`)
- Disease head: at least 10 of 15 outputs are different from each other

**Pass Criteria**
- `max(|f1 − f2|) < 1e-7` for 5 backbone runs
- `std(mc_probs across 15 runs) > 0.001` (at least one class shows variation)

---

### TC-37: Entropy Bounds — Uniform vs. Confident

| Field | Detail |
|---|---|
| **ID** | TC-37 |
| **Category** | Uncertainty |
| **Priority** | Medium |
| **Script** | Entropy formula: `H = -sum(p * log(p + 1e-10))` |

**Description**
For 5 classes, maximum entropy (uniform distribution) = `log(5) ≈ 1.609 nats`. Minimum entropy (completely certain) ≈ 0. The routing system uses `entropy_high = 1.0` nats as the ESCALATE trigger.

**Input**
- `p_uniform = [0.2, 0.2, 0.2, 0.2, 0.2]`
- `p_certain = [0.99, 0.0025, 0.0025, 0.0025, 0.0025]`

**Expected Output**
- `H(p_uniform) ≈ 1.609`
- `H(p_certain) ≈ 0.03` (near zero)
- `entropy_high = 1.0` nats is between these two extremes

**Pass Criteria**
- `1.60 < H(p_uniform) < 1.62`
- `H(p_certain) < 0.1`
- Routing threshold 1.0 nats correctly separates high-uncertainty from normal range

---

### TC-38: Uncertainty vs. Accuracy Correlation

| Field | Detail |
|---|---|
| **ID** | TC-38 |
| **Category** | Uncertainty |
| **Priority** | Medium |
| **Script** | `mc_dropout_uncertainty.py` |

**Description**
Higher MC Dropout entropy should correlate with lower accuracy. Samples in the top quartile of entropy should have lower accuracy than samples in the bottom quartile.

**Input**
MC Dropout uncertainty output file at `outputs_v3/uncertainty/`.

**Steps**
1. Load uncertainty results.
2. Split samples into low-entropy (Q1) and high-entropy (Q4) quartiles.
3. Compare accuracy in each quartile.

**Expected Output**
- Q4 (highest entropy) accuracy < Q1 (lowest entropy) accuracy
- Difference ≥ 10%

**Pass Criteria**
- `acc_Q1 > acc_Q4`
- `acc_Q1 - acc_Q4 >= 0.10`

---

## Section 10 — Fairness & LODO Analysis

### TC-39: LODO Validation — Results File Completeness

| Field | Detail |
|---|---|
| **ID** | TC-39 |
| **Category** | Fairness Analysis |
| **Priority** | High |
| **Script** | `run_paper_experiments.py` → `outputs_v3/lodo_results.json` |

**Description**
Leave-One-Dataset-Out (LODO) results for all 4 held-out sets must be present in `outputs_v3/lodo_results.json`. Reported values: APTOS 70.8%/0.829, MESSIDOR-2 61.6%/0.633, ODIR 51.8%/0.439, REFUGE2 88.8%/0.904.

**Input**
`outputs_v3/lodo_results.json`

**Steps**
1. Load file.
2. Verify all 4 entries exist.
3. Verify REFUGE2 is highest (88.8%) — it has only Normal+Glaucoma (2-class, easier).
4. Verify ODIR is lowest (51.8%) — all 5 classes, most heterogeneous.

**Expected Output**
- 4 entries: APTOS, MESSIDOR-2, ODIR, REFUGE2
- `REFUGE2_accuracy > APTOS_accuracy > MESSIDOR2_accuracy > ODIR_accuracy`
- All accuracy values in `(0.3, 1.0)`

**Pass Criteria**
- All 4 datasets present
- REFUGE2 accuracy ≥ 0.85
- ODIR accuracy ≤ 0.55
- Average weighted F1 ≈ 0.70 ± 0.05

---

### TC-40: Cross-Source Calibration Fairness

| Field | Detail |
|---|---|
| **ID** | TC-40 |
| **Category** | Fairness Analysis |
| **Priority** | Medium |
| **Script** | `fairness_analysis.py` |

**Description**
Calibration fairness requires that ECE per source does not differ by more than 0.05. A large differential would mean the model is well-calibrated for one source but overconfident for another.

**Input**
ECE computed per source from `outputs_v3/fairness/`.

**Steps**
1. Load per-source ECE values.
2. Compute `max_ECE − min_ECE` across 4 sources.

**Expected Output**
- No single source has ECE > 0.10
- Max-min ECE difference < 0.05 (acceptable calibration uniformity)

**Pass Criteria**
- All per-source ECE < 0.10
- `max(ECE) - min(ECE) < 0.05`

---

### TC-41: Ablation Study — Full Pipeline Outperforms Parts

| Field | Detail |
|---|---|
| **ID** | TC-41 |
| **Category** | Fairness / Ablation |
| **Priority** | High |
| **Script** | `outputs_v3/ablation_results.json` |

**Description**
The ablation study must confirm that DANN-v3 (full pipeline) outperforms all individual ablated variants. The full pipeline at 89.09% F1=0.879 must exceed: Base ViT (85.28%/0.843), DANN-only (84.73%/0.843), DANN+hard mining (85.89%/0.849), DANN+mixup (84.66%/0.821).

**Input**
`outputs_v3/ablation_results.json`

**Steps**
1. Load results for all 5 variants.
2. Rank by accuracy and macro F1.

**Expected Output**
- DANN-v3 full pipeline ranks 1st in both accuracy and macro F1
- All ablations below 87% accuracy
- Synergistic effect: full pipeline > any single component added to base DANN

**Pass Criteria**
- `dann_v3_full_acc > max(all_other_variant_acc)`
- `dann_v3_full_f1 > max(all_other_variant_f1)`
- No ablated variant exceeds 87% accuracy

---

## Section 11 — Gradio Demo & FastAPI Server

### TC-42: Gradio App Inference — End-to-End Output Fields

| Field | Detail |
|---|---|
| **ID** | TC-42 |
| **Category** | Demo Application |
| **Priority** | High |
| **Script** | `app.py` |

**Description**
The Gradio `predict()` function must return all required output fields: predicted disease, confidence, DR severity grade, probability bar chart, attention heatmap, uncertainty estimate, and routing tier.

**Input**
A valid JPEG fundus image uploaded via the Gradio interface (or passed directly to `predict()`).

**Steps**
1. Call the Gradio predict function with a test image.
2. Inspect returned tuple/dict.

**Expected Output**
- Disease name: one of `["Normal", "Diabetes/DR", "Glaucoma", "Cataract", "AMD"]`
- Confidence: float in `(0, 1)`
- Severity grade: integer in `{0, 1, 2, 3, 4}`
- Probability chart: matplotlib figure or PIL image (5 bars)
- Attention heatmap: `(224, 224)` overlay image
- Uncertainty: float ≥ 0
- Routing tier: one of `["AUTO-REPORT", "REVIEW", "ESCALATE"]`

**Pass Criteria**
- All 7 fields populated (no None values)
- Disease name in valid set
- Confidence in `(0, 1)`
- No exceptions raised

---

### TC-43: Model Loading Priority — DANN-v3 Preferred

| Field | Detail |
|---|---|
| **ID** | TC-43 |
| **Category** | Demo Application |
| **Priority** | High |
| **Script** | `app.py` lines 33–40 |

**Description**
The app checks 4 model paths in order: `dann_v3/best_model.pth` → `dann/best_model.pth` → `dann_v2/best_model.pth` → `best_model.pth`. The first existing file is used. DANN-v3 must be selected when present.

**Input**
All 4 paths, with only `dann_v3/best_model.pth` present.

**Steps**
Verify `MODEL_PATH` variable resolves to `dann_v3/best_model.pth`.

**Expected Output**
- `MODEL_PATH` ends with `dann_v3/best_model.pth`

**Input (fallback test)**
Rename `dann_v3` checkpoint, ensure `dann_v2` checkpoint exists.

**Expected Fallback**
- `MODEL_PATH` ends with `dann_v2/best_model.pth`
- No crash; correct fallback message printed

**Pass Criteria**
- Primary: `'dann_v3' in MODEL_PATH`
- Fallback: `'dann_v2' in MODEL_PATH` when v3 absent

---

### TC-44: FastAPI `/predict` Endpoint — JSON Response Schema

| Field | Detail |
|---|---|
| **ID** | TC-44 |
| **Category** | FastAPI Server |
| **Priority** | High |
| **Script** | `api/main.py` |

**Description**
The `POST /predict` endpoint accepts JSON with a base64-encoded image and returns a structured JSON response including disease, confidence, uncertainty, and class probabilities.

**Input**
```json
{
  "image": "<base64-encoded JPEG fundus image>"
}
```

**Steps**
1. Start FastAPI server: `uvicorn api.main:app --port 8000`.
2. POST to `http://localhost:8000/predict` with base64 image.
3. Inspect response JSON.

**Expected Output**
```json
{
  "disease": "Diabetes/DR",
  "confidence": 0.87,
  "severity": 2,
  "uncertainty": 0.12,
  "probabilities": {
    "Normal": 0.05, "Diabetes/DR": 0.87, "Glaucoma": 0.03,
    "Cataract": 0.03, "AMD": 0.02
  },
  "routing_tier": "AUTO-REPORT"
}
```

**Pass Criteria**
- HTTP status 200
- `disease` in valid class list
- `confidence` + all probabilities sum ≈ 1.0
- Response time < 2 seconds (GPU) or < 10 seconds (CPU)

---

### TC-45: FastAPI Health and Classes Endpoints

| Field | Detail |
|---|---|
| **ID** | TC-45 |
| **Category** | FastAPI Server |
| **Priority** | Medium |
| **Script** | `api/main.py` |

**Description**
The `GET /health` endpoint returns service status. The `GET /classes` endpoint returns the list of 5 class names. These are required for upstream load balancers and client initialization.

**Input**
Running FastAPI server at `http://localhost:8000`.

**Steps**
1. `GET /health` — check status field.
2. `GET /classes` — check returned list.
3. Access `GET /docs` — verify Swagger UI is available.

**Expected Output**
- `/health`: `{"status": "ok", ...}` with HTTP 200
- `/classes`: `["Normal", "Diabetes/DR", "Glaucoma", "Cataract", "AMD"]`
- `/docs`: Swagger UI HTML page (HTTP 200)

**Pass Criteria**
- `/health` returns HTTP 200 with `"status": "ok"`
- `/classes` returns list of exactly 5 class names in correct order
- `/docs` returns HTTP 200 (Swagger available)

---

## Appendix A — Test Environment Requirements

| Requirement | Specification |
|---|---|
| Python | 3.8+ |
| PyTorch | 2.0+ |
| timm | 0.9+ |
| faiss-gpu or faiss-cpu | 1.7+ |
| OpenCV | 4.5+ |
| scikit-learn | 1.0+ |
| gradio | 3.x or 4.x |
| fastapi + uvicorn | Latest stable |
| Production model | `outputs_v3/dann_v3/best_model.pth` |
| FAISS index | `outputs_v3/retrieval/index_flat_ip.faiss` |
| Metadata | `outputs_v3/retrieval/metadata.json` |
| Config files | `configs/temperature.json`, `configs/thresholds.json`, `configs/fundus_norm_stats_unified.json` |

---

## Appendix B — Expected Production Metrics (Reference)

| Metric | Value |
|---|---|
| Accuracy | 89.30% |
| Macro F1 | 0.886 |
| Macro AUC | 0.975 |
| ECE (post-calibration) | 0.037 |
| Temperature | 0.5657 |
| Cohen's Kappa | 0.809 |
| MCC | 0.810 |
| FAISS MAP | 0.921 |
| FAISS Recall@1 | 94.0% |
| RAD Combined Accuracy | 94.0% |
| AUTO-REPORT Fraction | 76.9% |
| AUTO-REPORT Accuracy | 96.8% |
| Error Catch Rate | 77.2% |
| LODO Average Accuracy | 68.2% |
| K-Fold Accuracy (5-fold) | 82.4% ± 1.9% |
| Normal F1 | 0.854 |
| DR F1 | 0.920 |
| Glaucoma F1 | 0.833 |
| Cataract F1 | 0.899 |
| AMD F1 | 0.895 |

---

## Appendix C — Per-Class Thresholds (Production)

| Class | Index | Threshold |
|---|---|---|
| Normal | 0 | 0.3990 |
| Diabetes/DR | 1 | 0.4173 |
| Glaucoma | 2 | 0.6745 |
| Cataract | 3 | 0.2888 |
| AMD | 4 | 0.3439 |

Glaucoma has the highest threshold because the model outputs lower raw probabilities for this minority class (308 training images), and the threshold is tuned to maximize binary F1 on the calibration set.

---

*Last updated: 2026-03-27 — covers DANN-v3 production model, full RAD pipeline, Gradio demo v6, and FastAPI server.*
