#!/usr/bin/env python3
"""
RETFound Backbone for RetinaSense
==================================
Provides a RETFound-based backbone (ViT-Base/16 pretrained via MAE on
1.6 million retinal images) as a drop-in replacement for the ImageNet-
pretrained ViT used in retinasense_v3.py.

RETFound (Retinal Foundation Model) was pretrained with masked
autoencoders on a large corpus of colour fundus photographs and OCT
scans.  Using it as the backbone gives the model domain-specific
features (vessel topology, optic-disc morphology, drusen texture) that
ImageNet weights cannot provide.

Weight download
---------------
Colour-fundus-photo weights are hosted on Hugging Face:
    Repository : rmaphoh/RETFound_MAE
    File       : RETFound_cfp_weights.pth

You can download them in one of two ways:

1. Programmatic (recommended):
       from retfound_backbone import setup_retfound
       path = setup_retfound()          # downloads ~350 MB on first call

2. Manual:
       pip install huggingface_hub
       huggingface-cli download rmaphoh/RETFound_MAE RETFound_cfp_weights.pth \\
           --local-dir ./weights

Reference
---------
Zhou et al., "A foundation model for generalizable disease detection
from retinal images", Nature 2023.
https://github.com/rmaphoh/RETFound_MAE

Usage with the training pipeline
---------------------------------
    from retfound_backbone import MultiTaskRetFound, setup_retfound

    weights_path = setup_retfound()          # or pass your own path
    model = MultiTaskRetFound(pretrained_path=weights_path).to(device)

The model exposes the same (disease_logits, severity_logits) forward
interface as MultiTaskViT in retinasense_v3.py, so the training loop,
LLRD optimiser, and evaluation code work without modification.
"""

import os
import re
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import timm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_HF_REPO = "rmaphoh/RETFound_MAE"
_HF_FILE = "RETFound_cfp_weights.pth"
_DEFAULT_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
_VIT_EMBED_DIM = 768   # ViT-Base CLS token dimension


# ===================================================================
# Weight download helper
# ===================================================================
def setup_retfound(
    save_dir: str = _DEFAULT_WEIGHTS_DIR,
    filename: str = _HF_FILE,
) -> str:
    """Download RETFound colour-fundus-photo weights from Hugging Face.

    Uses ``huggingface_hub.hf_hub_download`` so that repeated calls are
    no-ops (the hub client caches the file).

    Parameters
    ----------
    save_dir : str
        Local directory to store the weight file.  Created if absent.
    filename : str
        Name of the weight file on Hugging Face.

    Returns
    -------
    str
        Absolute path to the downloaded ``.pth`` file.

    Raises
    ------
    ImportError
        If ``huggingface_hub`` is not installed.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download RETFound weights.\n"
            "Install it with:  pip install huggingface_hub"
        )

    os.makedirs(save_dir, exist_ok=True)
    local_path = os.path.join(save_dir, filename)

    if os.path.isfile(local_path):
        logger.info("RETFound weights already present at %s", local_path)
        return local_path

    logger.info(
        "Downloading RETFound weights from %s/%s ...", _HF_REPO, filename
    )
    downloaded = hf_hub_download(
        repo_id=_HF_REPO,
        filename=filename,
        local_dir=save_dir,
        local_dir_use_symlinks=False,
    )
    logger.info("RETFound weights saved to %s", downloaded)
    return downloaded


# ===================================================================
# Key-mapping helpers  (RETFound MAE checkpoint -> timm ViT)
# ===================================================================
def _map_retfound_keys(mae_state_dict: dict) -> OrderedDict:
    """Translate a RETFound MAE encoder checkpoint into timm ViT keys.

    RETFound saves its encoder weights under a ``'model'`` (or
    ``'model_state_dict'``) top-level key.  Inside, the naming
    convention differs from timm's ``VisionTransformer``:

    RETFound key pattern              timm key pattern
    --------------------------------  --------------------------------
    encoder.patch_embed.*             patch_embed.*
    encoder.cls_token                 cls_token
    encoder.pos_embed                 pos_embed
    encoder.blocks.{i}.*             blocks.{i}.*
    encoder.norm.*                    norm.*
    fc_norm.*                         (skipped -- MAE head norm)
    decoder_*                         (skipped -- MAE decoder)
    mask_token                        (skipped)

    Some RETFound releases omit the ``encoder.`` prefix; both forms
    are handled.

    Parameters
    ----------
    mae_state_dict : dict
        The raw state dict loaded from the ``.pth`` file, *after*
        extracting the ``'model'`` sub-key if present.

    Returns
    -------
    OrderedDict
        State dict with keys compatible with
        ``timm.create_model('vit_base_patch16_224', num_classes=0)``.
    """
    mapped = OrderedDict()

    # Patterns to skip (decoder weights, mask token, MAE head norms)
    _skip_prefixes = ("decoder", "mask_token", "fc_norm", "head")

    for key, value in mae_state_dict.items():
        # Skip decoder / MAE-head parameters
        if any(key.startswith(p) for p in _skip_prefixes):
            continue

        new_key = key

        # Strip 'encoder.' prefix if present
        if new_key.startswith("encoder."):
            new_key = new_key[len("encoder."):]

        # Some checkpoints store blocks as 'encoder.blocks.N.*' which
        # after stripping becomes 'blocks.N.*' -- already correct for timm.

        # RETFound sometimes names the final LayerNorm 'norm.' which
        # matches timm, but occasionally uses 'ln_pre' or 'encoder_norm'.
        new_key = re.sub(r"^encoder_norm\.", "norm.", new_key)
        new_key = re.sub(r"^ln_pre\.", "norm.", new_key)

        mapped[new_key] = value

    return mapped


# ===================================================================
# Backbone factory
# ===================================================================
def create_retfound_model(pretrained_path: str = None) -> nn.Module:
    """Create a ViT-Base/16 backbone, optionally with RETFound weights.

    Parameters
    ----------
    pretrained_path : str or None
        Path to ``RETFound_cfp_weights.pth``.  When *None*, the model
        is initialised with ImageNet-pretrained timm weights (identical
        to the v3 baseline).

    Returns
    -------
    nn.Module
        A ``timm`` ``VisionTransformer`` with ``num_classes=0``
        (feature-extractor mode, returns CLS-token embeddings of
        dimension 768).
    """
    # Start from the same timm architecture used in retinasense_v3.py
    # so that LLRD, head structure, and image-size assumptions stay valid.
    backbone = timm.create_model(
        "vit_base_patch16_224",
        pretrained=(pretrained_path is None),  # ImageNet fallback
        num_classes=0,
    )

    if pretrained_path is not None:
        if not os.path.isfile(pretrained_path):
            raise FileNotFoundError(
                f"RETFound weights not found at {pretrained_path}. "
                f"Run  setup_retfound()  or download manually from "
                f"https://huggingface.co/{_HF_REPO}"
            )

        logger.info("Loading RETFound weights from %s", pretrained_path)
        raw_ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)

        # RETFound checkpoints wrap encoder weights under 'model' or
        # 'model_state_dict'.
        if "model" in raw_ckpt:
            raw_sd = raw_ckpt["model"]
        elif "model_state_dict" in raw_ckpt:
            raw_sd = raw_ckpt["model_state_dict"]
        elif "state_dict" in raw_ckpt:
            raw_sd = raw_ckpt["state_dict"]
        else:
            # Assume the file *is* the state dict directly
            raw_sd = raw_ckpt

        mapped_sd = _map_retfound_keys(raw_sd)

        # Load with strict=False: RETFound may lack timm's head.*
        # keys (we already set num_classes=0) and we deliberately
        # dropped the decoder.
        missing, unexpected = backbone.load_state_dict(mapped_sd, strict=False)

        # Filter out expected mismatches for clean logging
        expected_missing = {"head.weight", "head.bias"}
        real_missing = [k for k in missing if k not in expected_missing]

        if real_missing:
            logger.warning(
                "Keys in timm model but NOT in RETFound checkpoint (%d): %s",
                len(real_missing),
                real_missing[:10],
            )
        if unexpected:
            logger.warning(
                "Unexpected keys from RETFound checkpoint (%d): %s",
                len(unexpected),
                unexpected[:10],
            )

        n_loaded = len(mapped_sd) - len(unexpected)
        logger.info(
            "RETFound backbone loaded: %d parameters mapped, "
            "%d missing (expected), %d unexpected (skipped)",
            n_loaded,
            len(real_missing),
            len(unexpected),
        )

    return backbone


# ===================================================================
# Multi-task model with RETFound backbone
# ===================================================================
class MultiTaskRetFound(nn.Module):
    """ViT-Base/16 (RETFound) with disease + severity classification heads.

    Architecture mirrors ``MultiTaskViT`` from ``retinasense_v3.py`` so
    that the LLRD optimiser, Focal Loss, MixUp, and evaluation code
    work without changes.

    Parameters
    ----------
    n_disease : int
        Number of disease classes (default 5: Normal, DR, Glaucoma,
        Cataract, AMD).
    n_severity : int
        Number of DR severity grades (default 5: 0-4 APTOS scale).
    drop : float
        Dropout probability applied to the CLS embedding before the
        classification heads.
    pretrained_path : str or None
        Path to ``RETFound_cfp_weights.pth``.  Pass *None* to fall
        back to ImageNet-pretrained timm weights.
    """

    def __init__(
        self,
        n_disease: int = 5,
        n_severity: int = 5,
        drop: float = 0.3,
        pretrained_path: str = None,
    ):
        super().__init__()

        # --- Backbone ---
        self.backbone = create_retfound_model(pretrained_path=pretrained_path)
        feat = _VIT_EMBED_DIM  # 768

        # --- Shared dropout on CLS embedding ---
        self.drop = nn.Dropout(drop)

        # --- Disease classification head (5-class) ---
        # Same architecture as MultiTaskViT: 768 -> 512 -> 256 -> n_disease
        self.disease_head = nn.Sequential(
            nn.Linear(feat, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_disease),
        )

        # --- Severity grading head (5-class, APTOS DR grades 0-4) ---
        self.severity_head = nn.Sequential(
            nn.Linear(feat, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_severity),
        )

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of images, shape ``(B, 3, 224, 224)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(disease_logits, severity_logits)`` each of shape
            ``(B, n_classes)``.
        """
        f = self.backbone(x)       # (B, 768)
        f = self.drop(f)
        return self.disease_head(f), self.severity_head(f)


# ===================================================================
# LLRD helper (convenience re-export so callers do not have to touch
# retinasense_v3.py internals)
# ===================================================================
def get_retfound_optimizer_with_llrd(
    model: MultiTaskRetFound,
    base_lr: float = 3e-4,
    decay_factor: float = 0.75,
    weight_decay: float = 1e-4,
) -> torch.optim.AdamW:
    """Build an AdamW optimiser with Layer-wise Learning Rate Decay.

    Identical strategy to ``get_optimizer_with_llrd`` in retinasense_v3.py:
      - Head layers:              base_lr
      - Transformer blocks 11-0: base_lr * decay^(12-block_idx)
      - Patch/pos/cls embeddings: base_lr * decay^13

    Parameters
    ----------
    model : MultiTaskRetFound
        The model whose parameters will be optimised.
    base_lr : float
        Maximum learning rate (applied to classification heads).
    decay_factor : float
        Multiplicative factor per transformer block.
    weight_decay : float
        L2 regularisation coefficient.

    Returns
    -------
    torch.optim.AdamW
    """
    param_groups = []

    # 1. Classification heads (full LR)
    head_params = (
        list(model.disease_head.parameters())
        + list(model.severity_head.parameters())
        + list(model.drop.parameters())
    )
    param_groups.append({"params": head_params, "lr": base_lr})

    # 2. Transformer blocks (12 blocks, indexed 11 -> 0)
    blocks = model.backbone.blocks
    num_blocks = len(blocks)
    for block_idx in range(num_blocks - 1, -1, -1):
        distance = num_blocks - block_idx      # 1 for block[11], 12 for block[0]
        lr_i = base_lr * (decay_factor ** distance)
        param_groups.append({
            "params": list(blocks[block_idx].parameters()),
            "lr": lr_i,
        })

    # 3. Patch embed + positional embed + CLS token + final norm
    embed_lr = base_lr * (decay_factor ** (num_blocks + 1))
    embed_params = (
        list(model.backbone.patch_embed.parameters())
        + [model.backbone.cls_token, model.backbone.pos_embed]
        + list(model.backbone.norm.parameters())
    )
    param_groups.append({"params": embed_params, "lr": embed_lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    # Log LR spread
    lrs = [g["lr"] for g in param_groups]
    logger.info(
        "LLRD optimizer: %d groups | Head %.2e | Block[11] %.2e | "
        "Block[0] %.2e | Embed %.2e",
        len(param_groups), lrs[0], lrs[1], lrs[-2], lrs[-1],
    )

    return optimizer


# ===================================================================
# Quick sanity check
# ===================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Creating MultiTaskRetFound with ImageNet fallback weights ...")
    model = MultiTaskRetFound(pretrained_path=None)
    dummy = torch.randn(2, 3, 224, 224)
    d_out, s_out = model(dummy)
    print(f"  disease_logits  : {d_out.shape}")   # (2, 5)
    print(f"  severity_logits : {s_out.shape}")    # (2, 5)

    total = sum(p.numel() for p in model.parameters())
    print(f"  Total params    : {total:,}")

    opt = get_retfound_optimizer_with_llrd(model)
    print(f"  Optimizer groups: {len(opt.param_groups)}")

    print("\nTo load RETFound weights instead:")
    print("  path = setup_retfound()")
    print("  model = MultiTaskRetFound(pretrained_path=path)")
    print("\nDone.")
