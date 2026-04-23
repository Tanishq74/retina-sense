#!/usr/bin/env python3
"""
retfound_backbone.py  --  RETFound weight adapter for RetinaSense-ViT
=====================================================================

RETFound is a ViT-Base/16 pre-trained on 1.6 million retinal fundus images
via masked autoencoding (MAE).  This module downloads the public weights and
provides an adapter that loads them into the MultiTaskViT backbone used by
the RetinaSense project.

CLI usage
---------
    python retfound_backbone.py --setup      # download weights
    python retfound_backbone.py --verify     # verify weights loadable
    python retfound_backbone.py --compare    # compare param counts

Programmatic usage
------------------
    from retfound_backbone import load_retfound_into_vit
    load_retfound_into_vit(model, "weights/RETFound_cfp_weights.pth")
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RETFOUND_URL = (
    "https://huggingface.co/rmaphoh/RETFound/resolve/main/"
    "RETFound_cfp_weights.pth"
)
RETFOUND_HF_REPO = "rmaphoh/RETFound"
RETFOUND_HF_FILENAME = "RETFound_cfp_weights.pth"

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
WEIGHTS_PATH = WEIGHTS_DIR / "RETFound_cfp_weights.pth"

# Keys (or prefixes) in the RETFound checkpoint that belong to the
# MAE decoder or other components we do NOT want in our encoder backbone.
SKIP_PREFIXES: Tuple[str, ...] = (
    "head.",
    "fc_norm.",
    "decoder.",
    "decoder_embed.",
    "decoder_blocks.",
    "decoder_norm.",
    "decoder_pred.",
    "mask_token",
)

# Known source prefixes that may wrap the encoder keys.
STRIP_PREFIXES: Tuple[str, ...] = (
    "model.",
    "encoder.",
    "module.",
)

# Backbone sub-modules we expect in the timm ViT-Base/16 model.
BACKBONE_COMPONENTS: Set[str] = {
    "cls_token",
    "pos_embed",
    "patch_embed",
    "blocks",
    "norm",
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Callback for urllib.request.urlretrieve progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        bar_len = 40
        filled = int(bar_len * pct / 100)
        bar = "=" * filled + "-" * (bar_len - filled)
        mb_done = downloaded / 1e6
        mb_total = total_size / 1e6
        sys.stdout.write(
            f"\r  [{bar}] {pct:5.1f}%  ({mb_done:.1f}/{mb_total:.1f} MB)"
        )
    else:
        mb_done = downloaded / 1e6
        sys.stdout.write(f"\r  Downloaded {mb_done:.1f} MB ...")
    sys.stdout.flush()


def download_retfound(dest: Path = WEIGHTS_PATH, force: bool = False) -> Path:
    """
    Download RETFound colour-fundus-photo weights.

    Tries ``huggingface_hub.hf_hub_download`` first (resumable, cached),
    then falls back to ``requests`` with a progress bar, and finally
    ``urllib.request.urlretrieve``.

    Parameters
    ----------
    dest : Path
        Where to save the file.
    force : bool
        Re-download even if the file already exists.

    Returns
    -------
    Path
        The path to the downloaded weights file.
    """
    dest = Path(dest)
    if dest.exists() and not force:
        size_mb = dest.stat().st_size / 1e6
        print(f"[retfound] Weights already present at {dest} ({size_mb:.1f} MB)")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[retfound] Downloading RETFound weights to {dest} ...")

    # --- Strategy 1: huggingface_hub (preferred) ---
    try:
        from huggingface_hub import hf_hub_download

        print("  Using huggingface_hub.hf_hub_download ...")
        cached = hf_hub_download(
            repo_id=RETFOUND_HF_REPO,
            filename=RETFOUND_HF_FILENAME,
        )
        # hf_hub_download returns a cache path; symlink/copy to dest
        cached = Path(cached)
        if dest.exists():
            dest.unlink()
        try:
            os.symlink(cached, dest)
        except OSError:
            import shutil
            shutil.copy2(cached, dest)
        size_mb = dest.stat().st_size / 1e6
        print(f"\n[retfound] Download complete ({size_mb:.1f} MB)")
        return dest
    except ImportError:
        print("  huggingface_hub not installed, trying fallback ...")
    except Exception as exc:
        print(f"  huggingface_hub failed ({exc}), trying fallback ...")

    # --- Strategy 2: requests with progress ---
    try:
        import requests

        print("  Using requests ...")
        resp = requests.get(RETFOUND_URL, stream=True, timeout=30)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        chunk_size = 1 << 20  # 1 MB
        downloaded = 0
        tmp = dest.with_suffix(".pth.tmp")
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    bar_len = 40
                    filled = int(bar_len * pct / 100)
                    bar = "=" * filled + "-" * (bar_len - filled)
                    sys.stdout.write(
                        f"\r  [{bar}] {pct:5.1f}%  "
                        f"({downloaded/1e6:.1f}/{total/1e6:.1f} MB)"
                    )
                else:
                    sys.stdout.write(f"\r  Downloaded {downloaded/1e6:.1f} MB ...")
                sys.stdout.flush()
        tmp.rename(dest)
        size_mb = dest.stat().st_size / 1e6
        print(f"\n[retfound] Download complete ({size_mb:.1f} MB)")
        return dest
    except ImportError:
        print("  requests not installed, trying urllib ...")
    except Exception as exc:
        print(f"  requests failed ({exc}), trying urllib ...")

    # --- Strategy 3: urllib (always available) ---
    import urllib.request

    print("  Using urllib.request ...")
    tmp = dest.with_suffix(".pth.tmp")
    try:
        urllib.request.urlretrieve(RETFOUND_URL, str(tmp), reporthook=_progress_hook)
        tmp.rename(dest)
        size_mb = dest.stat().st_size / 1e6
        print(f"\n[retfound] Download complete ({size_mb:.1f} MB)")
        return dest
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"All download methods failed.  Last error: {exc}\n"
            f"You can manually download from:\n  {RETFOUND_URL}\n"
            f"and place the file at:\n  {dest}"
        ) from exc


# ---------------------------------------------------------------------------
# Key-mapping & position-embedding interpolation
# ---------------------------------------------------------------------------

def _strip_prefix(key: str) -> str:
    """Remove known wrapper prefixes from a checkpoint key."""
    for prefix in STRIP_PREFIXES:
        if key.startswith(prefix):
            key = key[len(prefix):]
            # Only strip one layer of prefix, then check again
            # (handles cases like model.encoder.blocks...)
            return _strip_prefix(key)
    return key


def _should_skip(key: str) -> bool:
    """Return True if the key belongs to a component we don't want."""
    for prefix in SKIP_PREFIXES:
        if key.startswith(prefix):
            return True
    return False


def _is_backbone_key(key: str) -> bool:
    """Return True if the key belongs to one of the expected backbone parts."""
    for comp in BACKBONE_COMPONENTS:
        if key == comp or key.startswith(comp + ".") or key.startswith(comp + "["):
            return True
    return False


def interpolate_pos_embed(
    pos_embed_ckpt: torch.Tensor,
    pos_embed_model: torch.Tensor,
    num_prefix_tokens: int = 1,
) -> torch.Tensor:
    """
    Interpolate position embeddings if the spatial resolution differs.

    Parameters
    ----------
    pos_embed_ckpt : Tensor   shape (1, N_ckpt, D)
    pos_embed_model : Tensor  shape (1, N_model, D)
    num_prefix_tokens : int   number of special tokens (CLS=1)

    Returns
    -------
    Tensor  shape (1, N_model, D)
    """
    if pos_embed_ckpt.shape == pos_embed_model.shape:
        return pos_embed_ckpt

    embed_dim = pos_embed_ckpt.shape[-1]
    N_ckpt = pos_embed_ckpt.shape[1] - num_prefix_tokens
    N_model = pos_embed_model.shape[1] - num_prefix_tokens

    # Separate CLS / prefix tokens from spatial tokens
    prefix_tokens = pos_embed_ckpt[:, :num_prefix_tokens, :]
    spatial_tokens = pos_embed_ckpt[:, num_prefix_tokens:, :]

    gs_ckpt = int(N_ckpt ** 0.5)
    gs_model = int(N_model ** 0.5)

    if gs_ckpt ** 2 != N_ckpt or gs_model ** 2 != N_model:
        print(
            f"  [warn] Non-square grid detected (ckpt={N_ckpt}, model={N_model}). "
            "Skipping pos_embed interpolation."
        )
        return pos_embed_model  # fall back to model's own init

    print(
        f"  Interpolating pos_embed: {gs_ckpt}x{gs_ckpt} -> {gs_model}x{gs_model}"
    )
    spatial_tokens = spatial_tokens.reshape(1, gs_ckpt, gs_ckpt, embed_dim).permute(
        0, 3, 1, 2
    )
    spatial_tokens = F.interpolate(
        spatial_tokens,
        size=(gs_model, gs_model),
        mode="bicubic",
        align_corners=False,
    )
    spatial_tokens = spatial_tokens.permute(0, 2, 3, 1).reshape(
        1, gs_model * gs_model, embed_dim
    )
    return torch.cat([prefix_tokens, spatial_tokens], dim=1)


# ---------------------------------------------------------------------------
# Main adapter function
# ---------------------------------------------------------------------------

def load_retfound_into_vit(
    model: nn.Module,
    weights_path: str | Path = WEIGHTS_PATH,
    strict: bool = False,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Load RETFound MAE pre-trained weights into a MultiTaskViT model.

    RETFound uses a MAE encoder, so the keys may differ from timm ViT:
    - RETFound keys may have ``encoder.`` or ``model.`` prefix
    - Head / decoder weights are ignored (we use our own heads)
    - Position embeddings are interpolated if spatial resolution differs

    Parameters
    ----------
    model : nn.Module
        A MultiTaskViT (or similar) with a ``backbone`` attribute that is
        a timm ``VisionTransformer``.
    weights_path : str or Path
        Path to ``RETFound_cfp_weights.pth``.
    strict : bool
        If True, raise on any missing / unexpected backbone keys.

    Returns
    -------
    loaded_keys : list[str]
        Keys successfully loaded into model.
    missing_keys : list[str]
        Model backbone keys not present in checkpoint.
    unexpected_keys : list[str]
        Checkpoint keys that were not used (after filtering).
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"RETFound weights not found at {weights_path}.\n"
            f"Run:  python retfound_backbone.py --setup"
        )

    print(f"[retfound] Loading checkpoint from {weights_path} ...")
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

    # Handle different checkpoint wrapping conventions
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]
    elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state_dict = ckpt["state_dict"]
    else:
        # Assume top-level is already the state dict
        state_dict = ckpt

    # --- Build remapped state dict ---
    remapped: Dict[str, torch.Tensor] = {}
    skipped_keys: List[str] = []

    for raw_key, value in state_dict.items():
        stripped = _strip_prefix(raw_key)

        if _should_skip(stripped):
            skipped_keys.append(raw_key)
            continue

        # Only keep backbone-relevant keys
        if not _is_backbone_key(stripped):
            skipped_keys.append(raw_key)
            continue

        target_key = f"backbone.{stripped}"
        remapped[target_key] = value

    print(f"  Checkpoint keys total     : {len(state_dict)}")
    print(f"  Mapped to backbone        : {len(remapped)}")
    print(f"  Skipped (head/decoder/etc): {len(skipped_keys)}")

    # --- Handle position embedding size mismatch ---
    pos_key = "backbone.pos_embed"
    if pos_key in remapped and hasattr(model, "backbone"):
        model_pos = model.state_dict().get(pos_key)
        if model_pos is not None and remapped[pos_key].shape != model_pos.shape:
            remapped[pos_key] = interpolate_pos_embed(
                remapped[pos_key],
                model_pos,
                num_prefix_tokens=1,
            )

    # --- Handle shape mismatches gracefully ---
    model_sd = model.state_dict()
    shape_mismatches: List[str] = []
    for key in list(remapped.keys()):
        if key in model_sd and remapped[key].shape != model_sd[key].shape:
            print(
                f"  [warn] Shape mismatch for {key}: "
                f"ckpt={list(remapped[key].shape)} vs "
                f"model={list(model_sd[key].shape)}  -- skipping"
            )
            shape_mismatches.append(key)
            del remapped[key]

    # --- Load ---
    result = model.load_state_dict(remapped, strict=False)

    # Compute our own tracking lists
    loaded_keys = [k for k in remapped if k not in result.unexpected_keys]
    missing_keys = [
        k for k in result.missing_keys if k.startswith("backbone.")
    ]
    unexpected_keys = result.unexpected_keys

    print(f"\n  === Load Summary ===")
    print(f"  Loaded into model         : {len(loaded_keys)}")
    print(f"  Missing backbone keys     : {len(missing_keys)}")
    print(f"  Shape mismatches (skipped): {len(shape_mismatches)}")
    if missing_keys:
        print(f"  Missing keys (first 10)   : {missing_keys[:10]}")
    if shape_mismatches:
        print(f"  Mismatch keys             : {shape_mismatches}")

    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(
            f"strict=True but found {len(missing_keys)} missing and "
            f"{len(unexpected_keys)} unexpected keys."
        )

    # Count parameters loaded
    params_loaded = sum(remapped[k].numel() for k in loaded_keys)
    params_total = sum(p.numel() for p in model.parameters())
    print(
        f"  Parameters loaded         : {params_loaded:,} / {params_total:,} "
        f"({params_loaded/params_total*100:.1f}%)"
    )

    return loaded_keys, missing_keys, unexpected_keys


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_weights(weights_path: Path = WEIGHTS_PATH) -> bool:
    """Check that weights exist, are loadable, and work in a forward pass."""
    print("=" * 60)
    print("RETFound Weight Verification")
    print("=" * 60)

    # 1. File exists
    if not weights_path.exists():
        print(f"[FAIL] Weights not found at {weights_path}")
        return False
    size_mb = weights_path.stat().st_size / 1e6
    print(f"[OK]   File exists: {weights_path} ({size_mb:.1f} MB)")

    # 2. Loadable as a torch checkpoint
    try:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
        else:
            sd = ckpt
        print(f"[OK]   Checkpoint loadable, contains {len(sd)} keys")
    except Exception as exc:
        print(f"[FAIL] Could not load checkpoint: {exc}")
        return False

    # 3. Build model and load weights
    try:
        import timm

        # Create a minimal MultiTaskViT-like model
        class _TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = timm.create_model(
                    "vit_base_patch16_224", pretrained=False, num_classes=0
                )
                self.disease_head = nn.Linear(768, 5)
                self.severity_head = nn.Linear(768, 5)

            def forward(self, x):
                f = self.backbone(x)
                return self.disease_head(f), self.severity_head(f)

        model = _TestModel()
        loaded, missing, unexpected = load_retfound_into_vit(
            model, weights_path, strict=False
        )
        print(f"[OK]   Weights loaded: {len(loaded)} keys")
        if missing:
            print(f"[WARN] {len(missing)} backbone keys not in checkpoint")
    except Exception as exc:
        print(f"[FAIL] Weight loading failed: {exc}")
        import traceback; traceback.print_exc()
        return False

    # 4. Forward pass
    try:
        model.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            disease_out, severity_out = model(dummy)
        assert disease_out.shape == (1, 5), f"Bad disease shape: {disease_out.shape}"
        assert severity_out.shape == (1, 5), f"Bad severity shape: {severity_out.shape}"
        print(f"[OK]   Forward pass successful")
        print(f"       disease_out  : shape={list(disease_out.shape)}")
        print(f"       severity_out : shape={list(severity_out.shape)}")
    except Exception as exc:
        print(f"[FAIL] Forward pass failed: {exc}")
        import traceback; traceback.print_exc()
        return False

    print("\n[PASS] All verification checks passed.")
    return True


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_imagenet_vs_retfound(weights_path: Path = WEIGHTS_PATH) -> None:
    """Compare parameter counts: ImageNet-pretrained ViT vs RETFound."""
    import timm

    print("=" * 60)
    print("Parameter Comparison: ImageNet ViT-B/16  vs  RETFound")
    print("=" * 60)

    # --- ImageNet model ---
    print("\n--- ImageNet-21k ViT-Base/16 (timm) ---")
    imagenet_model = timm.create_model(
        "vit_base_patch16_224", pretrained=False, num_classes=0
    )
    imagenet_sd = imagenet_model.state_dict()
    imagenet_params = sum(p.numel() for p in imagenet_model.parameters())
    print(f"  Total parameters : {imagenet_params:,}")
    print(f"  State dict keys  : {len(imagenet_sd)}")

    # --- RETFound checkpoint ---
    print("\n--- RETFound (MAE pre-trained) ---")
    if not weights_path.exists():
        print(f"  [ERROR] Weights not found at {weights_path}")
        print(f"  Run:  python retfound_backbone.py --setup")
        return

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
    elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    total_ckpt_params = sum(v.numel() for v in sd.values() if isinstance(v, torch.Tensor))
    print(f"  Total parameters : {total_ckpt_params:,}")
    print(f"  State dict keys  : {len(sd)}")

    # --- Breakdown by component ---
    # Map RETFound keys to stripped versions
    encoder_params = 0
    decoder_params = 0
    head_params = 0
    other_params = 0

    for raw_key, value in sd.items():
        if not isinstance(value, torch.Tensor):
            continue
        n = value.numel()
        stripped = _strip_prefix(raw_key)
        if _should_skip(stripped):
            if "decoder" in raw_key.lower():
                decoder_params += n
            else:
                head_params += n
        elif _is_backbone_key(stripped):
            encoder_params += n
        else:
            other_params += n

    print(f"\n  Encoder (backbone) params: {encoder_params:,}")
    print(f"  Decoder params           : {decoder_params:,}")
    print(f"  Head params              : {head_params:,}")
    print(f"  Other params             : {other_params:,}")

    # --- Key-level comparison ---
    print("\n--- Key Overlap ---")
    retfound_backbone_keys = set()
    for raw_key in sd:
        stripped = _strip_prefix(raw_key)
        if not _should_skip(stripped) and _is_backbone_key(stripped):
            retfound_backbone_keys.add(stripped)

    imagenet_keys = set(imagenet_sd.keys())
    common = retfound_backbone_keys & imagenet_keys
    only_retfound = retfound_backbone_keys - imagenet_keys
    only_imagenet = imagenet_keys - retfound_backbone_keys

    print(f"  Common keys              : {len(common)}")
    print(f"  Only in RETFound encoder : {len(only_retfound)}")
    print(f"  Only in ImageNet model   : {len(only_imagenet)}")

    if only_retfound:
        print(f"  RETFound-only (first 10) : {sorted(only_retfound)[:10]}")
    if only_imagenet:
        print(f"  ImageNet-only (first 10) : {sorted(only_imagenet)[:10]}")

    # --- Shape comparison for common keys ---
    shape_match = 0
    shape_mismatch = 0
    for key in sorted(common):
        ckpt_shape = sd.get(key, sd.get(f"model.{key}"))
        # Try to find the actual key in the checkpoint
        actual_val = None
        for raw_key, val in sd.items():
            if _strip_prefix(raw_key) == key:
                actual_val = val
                break
        if actual_val is not None and key in imagenet_sd:
            if actual_val.shape == imagenet_sd[key].shape:
                shape_match += 1
            else:
                shape_mismatch += 1
                print(
                    f"  [mismatch] {key}: "
                    f"RETFound={list(actual_val.shape)} vs "
                    f"ImageNet={list(imagenet_sd[key].shape)}"
                )

    print(f"\n  Shape matches            : {shape_match}")
    print(f"  Shape mismatches         : {shape_mismatch}")
    print(
        f"\n  Conclusion: RETFound encoder has {encoder_params:,} params, "
        f"ImageNet backbone has {imagenet_params:,} params."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RETFound weight downloader and adapter for RetinaSense-ViT"
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Download RETFound weights to ./weights/",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify weights exist, load correctly, and forward pass works",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare parameter counts between ImageNet ViT and RETFound",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if weights exist",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default=str(WEIGHTS_PATH),
        help=f"Custom path for weights file (default: {WEIGHTS_PATH})",
    )

    args = parser.parse_args()
    path = Path(args.weights_path)

    if not (args.setup or args.verify or args.compare):
        parser.print_help()
        sys.exit(1)

    if args.setup:
        try:
            download_retfound(dest=path, force=args.force)
        except Exception as exc:
            print(f"\n[ERROR] Download failed: {exc}", file=sys.stderr)
            sys.exit(1)

    if args.verify:
        ok = verify_weights(weights_path=path)
        if not ok:
            sys.exit(1)

    if args.compare:
        compare_imagenet_vs_retfound(weights_path=path)


if __name__ == "__main__":
    main()
