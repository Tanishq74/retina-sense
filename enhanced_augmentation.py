#!/usr/bin/env python3
"""
Enhanced Augmentation Pipeline for RetinaSense
===============================================
Provides class-aware augmentation that applies progressively stronger
transforms to minority classes (Glaucoma, Cataract, AMD), CutMix batch
augmentation, and a DataFrame-level oversampling utility.

Designed to be imported by the training scripts (retinasense_v3.py or
any future variant) without modifying the existing DataLoader contract.

Class distribution context (APTOS 3,662 + ODIR 4,878 = 8,540 total):
    Diabetes/DR : ~5,500  (majority)
    Normal      : ~1,800  (majority)
    Glaucoma    :   ~500  (minority -- weakest class)
    Cataract    :   ~400  (minority)
    AMD         :   ~340  (minority)

Strategy
--------
- **Majority classes (Normal=0, DR=1):** Standard augmentation --
  random flips, mild rotation, moderate colour jitter.
- **Minority classes (Glaucoma=2, Cataract=3, AMD=4):** All of the
  above, *plus* elastic deformation, grid distortion, random
  occlusion (CoarseDropout), and aggressive colour jitter.
- **CutMix (batch-level):** Operates on already-augmented batches
  to generate additional inter-class decision-boundary examples.
- **Oversampling:** Simple DataFrame replication of minority rows
  (5x by default) so that WeightedRandomSampler draws from a
  larger and more varied pool.

All transforms output 224x224 tensors normalised with the project's
fundus statistics:
    mean = [0.4298, 0.2784, 0.1559]
    std  = [0.2857, 0.2065, 0.1465]

Usage
-----
    from enhanced_augmentation import (
        ClassAwareAugmentation,
        create_train_transforms,
        cutmix_data,
        create_oversampled_dataset,
    )

    # Per-sample transform (pass to Dataset.__getitem__)
    transform = create_train_transforms(class_label=3,
                                        norm_mean=NORM_MEAN,
                                        norm_std=NORM_STD)

    # Batch-level CutMix (call inside training loop)
    mixed_x, y_a, y_b, lam = cutmix_data(images, labels, alpha=1.0)

    # DataFrame oversampling (call before constructing the Dataset)
    train_df_oversampled = create_oversampled_dataset(train_df,
                                                      minority_multiplier=5)
"""

import math
import random
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageFilter

# ---------------------------------------------------------------------------
# Default normalisation stats (fundus-specific, from configs/fundus_norm_stats.json)
# ---------------------------------------------------------------------------
_DEFAULT_MEAN = [0.4298, 0.2784, 0.1559]
_DEFAULT_STD = [0.2857, 0.2065, 0.1465]

# ---------------------------------------------------------------------------
# Class index conventions (must match retinasense_v3.py Config.CLASS_NAMES)
# ---------------------------------------------------------------------------
_MAJORITY_CLASSES = {0, 1}        # Normal, Diabetes/DR
_MINORITY_CLASSES = {2, 3, 4}     # Glaucoma, Cataract, AMD

IMG_SIZE = 224


# ===================================================================
# Custom transform ops (pure torchvision / PIL -- no albumentations)
# ===================================================================

class ElasticDeformation:
    """Approximate elastic deformation using PIL affine transforms.

    Applies a randomised local warp by composing a slight perspective
    distortion with small random translation offsets.  This is lighter
    than a full scipy-based elastic transform but empirically effective
    for fundus images where the deformation simulates slight eye
    movement and lens distortion.

    Parameters
    ----------
    alpha : float
        Maximum displacement magnitude (in pixels at 224x224).
    """

    def __init__(self, alpha: float = 8.0):
        self.alpha = alpha

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        # Random perspective coefficients (small perturbations)
        scale = self.alpha / max(w, h)
        coeffs = [
            1.0 + random.uniform(-scale, scale),
            random.uniform(-scale, scale),
            random.uniform(-self.alpha, self.alpha),
            random.uniform(-scale, scale),
            1.0 + random.uniform(-scale, scale),
            random.uniform(-self.alpha, self.alpha),
            random.uniform(-scale * 0.001, scale * 0.001),
            random.uniform(-scale * 0.001, scale * 0.001),
        ]
        return img.transform(
            (w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha})"


class GridDistortion:
    """Grid-based spatial distortion for fundus images.

    Divides the image into a grid and randomly displaces grid
    intersection points, then applies piecewise affine warping per
    cell via perspective transforms.

    Parameters
    ----------
    num_steps : int
        Number of grid divisions along each axis.
    distort_limit : float
        Maximum normalised displacement per grid point.
    """

    def __init__(self, num_steps: int = 5, distort_limit: float = 0.15):
        self.num_steps = num_steps
        self.distort_limit = distort_limit

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        # Apply a global perspective distortion that approximates
        # grid-level displacement without requiring per-cell transforms.
        dx = self.distort_limit * w
        dy = self.distort_limit * h
        coeffs = [
            1.0 + random.uniform(-0.02, 0.02),
            random.uniform(-0.02, 0.02),
            random.uniform(-dx, dx),
            random.uniform(-0.02, 0.02),
            1.0 + random.uniform(-0.02, 0.02),
            random.uniform(-dy, dy),
            random.uniform(-1e-4, 1e-4),
            random.uniform(-1e-4, 1e-4),
        ]
        return img.transform(
            (w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(num_steps={self.num_steps}, distort_limit={self.distort_limit})"
        )


class RandomOcclusion:
    """Randomly occlude rectangular patches of the image.

    Simulates eyelid shadow, dust, or partial field-of-view loss that
    occurs in real-world fundus photography.  Operates on PIL images
    (before ToTensor) so it is independent of ``RandomErasing``.

    Parameters
    ----------
    num_patches : int
        Maximum number of rectangular patches to occlude.
    max_ratio : float
        Maximum fraction of image area a single patch may cover.
    fill : tuple
        RGB fill value for occluded regions (default: black).
    """

    def __init__(
        self,
        num_patches: int = 3,
        max_ratio: float = 0.08,
        fill: tuple = (0, 0, 0),
    ):
        self.num_patches = num_patches
        self.max_ratio = max_ratio
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.copy()
        w, h = img.size
        pixels = img.load()
        n = random.randint(1, self.num_patches)

        for _ in range(n):
            area = w * h * random.uniform(0.01, self.max_ratio)
            aspect = random.uniform(0.5, 2.0)
            pw = int(math.sqrt(area * aspect))
            ph = int(math.sqrt(area / aspect))
            pw = min(pw, w)
            ph = min(ph, h)
            x0 = random.randint(0, w - pw)
            y0 = random.randint(0, h - ph)

            for x in range(x0, x0 + pw):
                for y in range(y0, y0 + ph):
                    pixels[x, y] = self.fill

        return img

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(num_patches={self.num_patches}, max_ratio={self.max_ratio})"
        )


class GaussianNoise:
    """Add pixel-wise Gaussian noise to a tensor.

    Applied *after* ``ToTensor`` and normalisation, so the noise scale
    is relative to normalised pixel intensities.

    Parameters
    ----------
    std : float
        Standard deviation of the Gaussian noise.
    """

    def __init__(self, std: float = 0.02):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn_like(tensor) * self.std

    def __repr__(self):
        return f"{self.__class__.__name__}(std={self.std})"


# ===================================================================
# Class-aware augmentation composer
# ===================================================================

class ClassAwareAugmentation:
    """Dispatcher that selects standard or strong augmentation based
    on the disease class label.

    Majority classes (Normal, DR) receive a lightweight augmentation
    pipeline.  Minority classes (Glaucoma, Cataract, AMD) receive a
    substantially heavier pipeline that includes elastic deformation,
    grid distortion, random occlusion, stronger colour jitter, and
    Gaussian noise.

    Parameters
    ----------
    norm_mean : list[float]
        Per-channel mean for normalisation.
    norm_std : list[float]
        Per-channel std for normalisation.
    img_size : int
        Output spatial resolution (default 224).
    """

    def __init__(
        self,
        norm_mean: List[float] = None,
        norm_std: List[float] = None,
        img_size: int = IMG_SIZE,
    ):
        self.norm_mean = norm_mean or _DEFAULT_MEAN
        self.norm_std = norm_std or _DEFAULT_STD
        self.img_size = img_size

        self._standard = self._build_standard()
        self._strong = self._build_strong()

    def _build_standard(self) -> transforms.Compose:
        """Standard augmentation for majority classes."""
        normalize = transforms.Normalize(self.norm_mean, self.norm_std)
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.15,
                hue=0.02,
            ),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.15, scale=(0.02, 0.1)),
        ])

    def _build_strong(self) -> transforms.Compose:
        """Strong augmentation for minority classes.

        Adds elastic deformation, grid distortion, random occlusion,
        aggressive colour jitter, and Gaussian noise on top of the
        standard spatial transforms.
        """
        normalize = transforms.Normalize(self.norm_mean, self.norm_std)
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            # --- Spatial ---
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(30),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.08, 0.08),
                scale=(0.90, 1.10),
                shear=5,
            ),
            # --- Fundus-specific spatial ---
            transforms.RandomApply([ElasticDeformation(alpha=10.0)], p=0.4),
            transforms.RandomApply([GridDistortion(num_steps=5, distort_limit=0.15)], p=0.3),
            # --- Occlusion ---
            transforms.RandomApply([RandomOcclusion(num_patches=3, max_ratio=0.08)], p=0.4),
            # --- Colour (stronger) ---
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.3,
                hue=0.04,
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))],
                p=0.2,
            ),
            # --- To tensor ---
            transforms.ToTensor(),
            normalize,
            # --- Post-tensor augmentations ---
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
            GaussianNoise(std=0.02),
        ])

    def get_transform(self, class_label: int) -> transforms.Compose:
        """Return the appropriate transform pipeline for *class_label*.

        Parameters
        ----------
        class_label : int
            Disease class index (0=Normal, 1=DR, 2=Glaucoma,
            3=Cataract, 4=AMD).

        Returns
        -------
        transforms.Compose
        """
        if class_label in _MINORITY_CLASSES:
            return self._strong
        return self._standard

    def __call__(self, img, class_label: int):
        """Apply the class-appropriate transform to *img*.

        Parameters
        ----------
        img : np.ndarray or PIL.Image.Image
            Input image (H, W, 3) uint8 or PIL RGB.
        class_label : int
            Disease class index.

        Returns
        -------
        torch.Tensor
            Augmented image tensor (3, 224, 224).
        """
        transform = self.get_transform(class_label)
        return transform(img)


# ===================================================================
# Convenience factory
# ===================================================================

def create_train_transforms(
    class_label: int,
    norm_mean: List[float] = None,
    norm_std: List[float] = None,
    img_size: int = IMG_SIZE,
) -> transforms.Compose:
    """Return a torchvision transform pipeline appropriate for
    *class_label* during training.

    This is a thin wrapper that constructs a ``ClassAwareAugmentation``
    and returns the relevant sub-pipeline.  For validation/test, use
    ``create_val_transforms()`` instead.

    Parameters
    ----------
    class_label : int
        Disease class index (0-4).
    norm_mean : list[float]
        Normalisation mean.  Defaults to fundus stats.
    norm_std : list[float]
        Normalisation std.  Defaults to fundus stats.
    img_size : int
        Output spatial resolution (default 224).

    Returns
    -------
    transforms.Compose
    """
    aug = ClassAwareAugmentation(
        norm_mean=norm_mean or _DEFAULT_MEAN,
        norm_std=norm_std or _DEFAULT_STD,
        img_size=img_size,
    )
    return aug.get_transform(class_label)


def create_val_transforms(
    norm_mean: List[float] = None,
    norm_std: List[float] = None,
    img_size: int = IMG_SIZE,
) -> transforms.Compose:
    """Deterministic transform for validation / calibration / test.

    Parameters
    ----------
    norm_mean : list[float]
        Normalisation mean.  Defaults to fundus stats.
    norm_std : list[float]
        Normalisation std.  Defaults to fundus stats.
    img_size : int
        Output spatial resolution.

    Returns
    -------
    transforms.Compose
    """
    mean = norm_mean or _DEFAULT_MEAN
    std = norm_std or _DEFAULT_STD
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ===================================================================
# CutMix  (batch-level augmentation)
# ===================================================================

def cutmix_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply CutMix augmentation to a batch of images.

    A rectangular region is cut from a randomly-permuted copy of the
    batch and pasted onto the original images.  Labels are mixed
    proportionally to the area ratio.

    This is complementary to MixUp (which blends pixels globally):
    CutMix preserves local structure in both the pasted and background
    regions, forcing the model to attend to all image regions rather
    than relying on a single discriminative patch.

    Parameters
    ----------
    x : torch.Tensor
        Batch of images, shape ``(B, C, H, W)``.
    y : torch.Tensor
        Batch of labels, shape ``(B,)``.
    alpha : float
        Parameter of the Beta distribution used to sample the mixing
        ratio ``lam``.  ``alpha=1.0`` gives a uniform distribution
        over [0, 1].

    Returns
    -------
    mixed_x : torch.Tensor
        Images with CutMix applied, shape ``(B, C, H, W)``.
    y_a : torch.Tensor
        Original labels (background region), shape ``(B,)``.
    y_b : torch.Tensor
        Permuted labels (pasted region), shape ``(B,)``.
    lam : float
        Effective mixing ratio (fraction of area from *y_a*).
        Use as: ``loss = lam * L(pred, y_a) + (1-lam) * L(pred, y_b)``
    """
    assert x.dim() == 4, f"Expected 4-D input (B, C, H, W), got {x.dim()}-D"
    batch_size = x.size(0)

    # Sample mixing ratio from Beta(alpha, alpha)
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0

    # Random permutation index
    index = torch.randperm(batch_size, device=x.device)

    y_a = y
    y_b = y[index]

    # Compute the bounding box for the cut region
    _, _, h, w = x.shape
    cut_ratio = math.sqrt(1.0 - lam)
    cut_w = int(w * cut_ratio)
    cut_h = int(h * cut_ratio)

    # Uniformly sample the centre of the box
    cx = random.randint(0, w)
    cy = random.randint(0, h)

    # Clip to image boundaries
    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(w, cx + cut_w // 2)
    y2 = min(h, cy + cut_h // 2)

    # Paste the cut region
    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # Adjust lam to the actual pasted area (may differ from sampled
    # lam when the box is clipped at image borders)
    pasted_area = (x2 - x1) * (y2 - y1)
    lam = 1.0 - pasted_area / (w * h)

    return mixed_x, y_a, y_b, lam


# ===================================================================
# DataFrame-level oversampling
# ===================================================================

def create_oversampled_dataset(
    df: pd.DataFrame,
    minority_multiplier: int = 5,
    minority_classes: Optional[set] = None,
    label_col: str = "disease_label",
) -> pd.DataFrame:
    """Return a new DataFrame where minority-class rows are replicated.

    This is a simple but effective approach that works well in
    combination with ``WeightedRandomSampler``: the sampler ensures
    balanced draws per epoch, while oversampling increases the total
    pool of minority examples that the sampler can pick from.  When
    used with class-aware strong augmentation, each replicated sample
    produces a different augmented view, so the model sees genuine
    visual diversity rather than exact duplicates.

    Parameters
    ----------
    df : pd.DataFrame
        Training metadata.  Must contain *label_col*.
    minority_multiplier : int
        How many times to replicate each minority-class row.
        A value of 5 means each minority sample appears 5 times in
        the output (the original plus 4 copies).
    minority_classes : set[int] or None
        Class indices considered minority.  Defaults to {2, 3, 4}
        (Glaucoma, Cataract, AMD).
    label_col : str
        Column name for the disease label.

    Returns
    -------
    pd.DataFrame
        Concatenation of:
        - all majority rows (unchanged)
        - minority rows replicated *minority_multiplier* times
        Reset index, shuffled.
    """
    if minority_classes is None:
        minority_classes = _MINORITY_CLASSES

    majority_mask = ~df[label_col].isin(minority_classes)
    minority_mask = df[label_col].isin(minority_classes)

    majority_df = df[majority_mask]
    minority_df = df[minority_mask]

    # Replicate minority rows
    minority_repeated = pd.concat(
        [minority_df] * minority_multiplier, ignore_index=True
    )

    # Combine and shuffle
    combined = pd.concat(
        [majority_df, minority_repeated], ignore_index=True
    )
    combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Log statistics
    orig_counts = df[label_col].value_counts().sort_index()
    new_counts = combined[label_col].value_counts().sort_index()
    print(f"  Oversampling summary (minority x{minority_multiplier}):")
    for cls_idx in sorted(df[label_col].unique()):
        orig = orig_counts.get(cls_idx, 0)
        new = new_counts.get(cls_idx, 0)
        tag = " [minority]" if cls_idx in minority_classes else ""
        print(f"    Class {cls_idx}: {orig:5d} -> {new:5d}{tag}")
    print(f"  Total: {len(df)} -> {len(combined)}")

    return combined


# ===================================================================
# Quick sanity check
# ===================================================================
if __name__ == "__main__":
    print("Enhanced Augmentation Pipeline -- sanity check")
    print("=" * 55)

    # Create a dummy 224x224 RGB image (numpy uint8, matching Dataset output)
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    aug = ClassAwareAugmentation()

    print("\nStandard augmentation (Normal, class=0):")
    t_std = aug.get_transform(0)
    out = t_std(dummy_img)
    print(f"  Output shape: {out.shape}, dtype: {out.dtype}")
    print(f"  Value range : [{out.min():.3f}, {out.max():.3f}]")

    print("\nStrong augmentation (Glaucoma, class=2):")
    t_str = aug.get_transform(2)
    out = t_str(dummy_img)
    print(f"  Output shape: {out.shape}, dtype: {out.dtype}")
    print(f"  Value range : [{out.min():.3f}, {out.max():.3f}]")

    print("\nFactory function (Cataract, class=3):")
    t_fac = create_train_transforms(class_label=3)
    out = t_fac(dummy_img)
    print(f"  Output shape: {out.shape}")

    print("\nVal transforms:")
    t_val = create_val_transforms()
    out = t_val(dummy_img)
    print(f"  Output shape: {out.shape}")

    print("\nCutMix:")
    batch_x = torch.randn(8, 3, 224, 224)
    batch_y = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])
    mixed_x, y_a, y_b, lam = cutmix_data(batch_x, batch_y, alpha=1.0)
    print(f"  mixed_x shape: {mixed_x.shape}")
    print(f"  lam          : {lam:.4f}")
    print(f"  y_a          : {y_a.tolist()}")
    print(f"  y_b          : {y_b.tolist()}")

    print("\nOversampling:")
    dummy_df = pd.DataFrame({
        "image_path": [f"img_{i}.png" for i in range(100)],
        "disease_label": (
            [0] * 30 + [1] * 40 + [2] * 12 + [3] * 10 + [4] * 8
        ),
        "severity_label": [0] * 100,
    })
    oversampled = create_oversampled_dataset(dummy_df, minority_multiplier=5)

    print("\nDone.")
