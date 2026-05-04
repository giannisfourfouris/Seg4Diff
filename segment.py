"""
SD3 Image Segmentation Script (based on Seg4Diff framework)

Segments images using Stable Diffusion 3's cross-attention maps,
following the Seg4Diff pipeline. Supports both open-vocabulary semantic
segmentation (with user-provided class labels) and unsupervised
segmentation (automatic discovery via KL-based mask merging).

Optionally loads LoRA weights and generates side-by-side comparisons
showing how LoRA fine-tuning affects the segmentation.

Usage:
    # Open-vocabulary segmentation with class labels
    python segment.py --image photo.jpg --classes "cat" "grass" "sky"

    # Unsupervised segmentation (no class labels needed)
    python segment.py --image photo.jpg --unsupervised

    # With LoRA weights for comparison
    python segment.py --image photo.jpg --classes "cat" "grass" "sky" \
        --lora_weights lora_weights/lora_weights.pth

    # Batch-process a folder of images
    python segment.py --image_dir ./my_photos --classes "person" "car" "tree"

Output structure:
    outputs/segmentation/
    └── photo_name/
        ├── base/
        │   ├── masks/          (per-class grayscale masks)
        │   ├── heatmaps/       (per-class jet heatmap plots)
        │   ├── argmax.png      (grayscale argmax segmentation)
        │   ├── argmax_color.png(color-coded segmentation)
        │   ├── overlay.png     (blended overlay on original)
        │   └── summary.png     (3-panel: original + segmap + overlay)
        ├── lora/               (same structure, only if --lora_weights)
        │   └── ...
        └── comparison.png      (base vs LoRA side-by-side)
"""

import argparse
import math
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

from diffusers import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor
from peft import LoraConfig, set_peft_model_state_dict


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Segment images with SD3 cross-attention (Seg4Diff)"
    )

    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=str, help="Path to a single image")
    input_group.add_argument("--image_dir", type=str, help="Directory of images to segment")

    # Segmentation mode
    parser.add_argument(
        "--classes", type=str, nargs="+", default=None,
        help="Class labels for open-vocabulary segmentation (e.g. cat grass sky)",
    )
    parser.add_argument(
        "--unsupervised", action="store_true",
        help="Use unsupervised segmentation (KL-based mask merging, no class labels)",
    )

    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs/segmentation")

    # Model
    parser.add_argument(
        "--model_id", type=str,
        default="stabilityai/stable-diffusion-3-medium-diffusers",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device (default: cuda). Use cuda:1 for second GPU.",
    )
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no_fp16", action="store_true")

    # Segmentation parameters
    parser.add_argument(
        "--noise_steps", type=int, default=8,
        help="Noise timestep index for feature extraction (default: 8, matching eval_unsup.yaml)",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=28,
        help="Total scheduler inference steps (default: 28)",
    )
    parser.add_argument(
        "--attention_layers", type=int, nargs="*",
        default=[9],
        help="Transformer block indices to cache attention from (default: 9, the expert layer)",
    )
    parser.add_argument(
        "--norm_before_merge", action="store_true", default=False,
        help="Normalize attention before merging tokens (default: False, matching eval_ovss.yaml)",
    )
    parser.add_argument(
        "--norm_after_merge", action="store_true", default=False,
        help="Normalize attention after merging tokens (default: False, matching eval_ovss.yaml)",
    )
    parser.add_argument(
        "--kl_threshold", type=float, default=0.035,
        help="KL divergence threshold for unsupervised mask merging (default: 0.035, matching eval_unsup.yaml)",
    )
    parser.add_argument(
        "--output_power", type=float, default=1.0,
        help="Power exponent for att2mask normalization in unsupervised mode (default: 1.0, matching eval_unsup.yaml)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resolution", type=int, default=1024,
        help="Input image resolution (default: 1024). Also supports 512.",
    )

    # LoRA
    parser.add_argument("--lora_weights", type=str, default=None)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_targets", type=str, default=None)
    parser.add_argument("--lora_layers", type=int, nargs="*", default=None)
    parser.add_argument(
        "--lora_weight_prefix", type=str, default="backbone.transformer.",
    )

    args = parser.parse_args()
    if args.no_fp16:
        args.fp16 = False
    if args.unsupervised:
        if "--norm_before_merge" not in sys.argv and "--norm_after_merge" not in sys.argv:
            args.norm_before_merge = True
            args.norm_after_merge = True
    if not args.unsupervised and args.classes is None:
        parser.error("Provide --classes for open-vocabulary mode, or use --unsupervised")
    return args


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def load_pipeline(model_id, fp16=True, device="cuda"):
    print(f"Loading SD3 pipeline from: {model_id}")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        text_encoder_3=None,
        torch_dtype=torch.float16 if fp16 else None,
    )
    pipe.to(device)
    print("Pipeline loaded.")
    return pipe


def setup_attention_caching(pipe, attention_layers, softmax=True):
    """Enable attention caching on selected transformer blocks (mirrors sd3_backbone.py)."""
    for l, blk in enumerate(pipe.transformer.transformer_blocks):
        blk.layer_id = l
        blk.attn.processor.attn_cache = None
        blk.attn.processor.feat_cache = None
        if l in attention_layers:
            blk.attn.processor.attn_cache = []
            blk.attn.processor.q = 171
            blk.attn.processor.softmax = softmax
            blk.attn.processor.head = None
            blk.attn.processor.keep_head = False
            blk.attn.processor.max_head = False
            blk.attn.processor.softmax_i2t_only = False


def clear_attention_cache(pipe):
    for blk in pipe.transformer.transformer_blocks:
        if blk.attn.processor.attn_cache is not None:
            blk.attn.processor.attn_cache = []


def collect_attention_cache(pipe):
    out = []
    for blk in pipe.transformer.transformer_blocks:
        cache = blk.attn.processor.attn_cache
        if cache is not None and not isinstance(cache, list):
            out.append(cache)
        elif isinstance(cache, list) and len(cache) > 0:
            out.append(cache[0] if len(cache) == 1 else torch.stack(cache))
    if len(out) == 0:
        return None
    return torch.stack(out, dim=1)  # [B, num_layers, n_img_patches, text_len]


def apply_lora(pipe, lora_weights_path, lora_rank=16, target_modules=None,
               lora_layers=None, weight_prefix="backbone.transformer."):
    if target_modules is None:
        target_modules = ["attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0"]

    if lora_layers is not None:
        target_modules = [
            f"transformer_blocks.{block}.{module}"
            for block in lora_layers for module in target_modules
        ]

    cfg = LoraConfig(
        r=lora_rank, lora_alpha=lora_rank,
        init_lora_weights="gaussian", target_modules=target_modules,
    )
    pipe.transformer.add_adapter(cfg)
    for name, param in pipe.transformer.named_parameters():
        param.requires_grad = "lora" in name

    def preprocess_state_dict(sd, prefix):
        new = {}
        for k, v in sd.items():
            if k.startswith(prefix):
                k = k[len(prefix):]
            k = k.replace(".default.", ".")
            new[k] = v
        return new

    print(f"Loading LoRA weights from: {lora_weights_path}")
    try:
        sd = torch.load(lora_weights_path, map_location="cpu", weights_only=False)
    except Exception:
        from safetensors.torch import load_file
        sd = load_file(lora_weights_path)
    sd_to_load = preprocess_state_dict(sd.get("model", sd) if isinstance(sd, dict) else sd, weight_prefix)
    outcome = set_peft_model_state_dict(pipe.transformer, sd_to_load)
    print(f"LoRA loaded. Unexpected keys: {outcome.unexpected_keys[:5] if outcome.unexpected_keys else 'none'}")
    return pipe


# ---------------------------------------------------------------------------
# Image → latent → noisy latent → transformer forward  (mirrors sd3_backbone)
# ---------------------------------------------------------------------------

def img_to_latents(x, vae):
    x = 2.0 * x - 1.0
    posterior = vae.encode(x).latent_dist
    return (posterior.mean - vae.config.shift_factor) * vae.config.scaling_factor


@torch.no_grad()
def extract_attention(pipe, image_tensor, prompt, noise_steps, num_inference_steps, seed=42):
    """
    Run one noisy forward pass through the SD3 transformer and return
    the cached image-to-text attention maps.

    Returns:
        attn_cache: Tensor [B, num_layers, 4096, text_len]
    """
    device = image_tensor.device

    latent = img_to_latents(image_tensor, pipe.vae).to(device)

    noise = randn_tensor(latent.shape, device=device,
                         generator=torch.Generator(device=device).manual_seed(seed))

    timesteps, _ = retrieve_timesteps(pipe.scheduler, num_inference_steps, device)
    t = timesteps[-noise_steps]
    noisy_latent = pipe.scheduler.scale_noise(latent, t.unsqueeze(0), noise)
    timestep = t.expand(noisy_latent.shape[0])

    prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
        prompt=prompt, prompt_2=None, prompt_3=None,
        do_classifier_free_guidance=False, device=device,
        max_sequence_length=77,
    )

    clear_attention_cache(pipe)

    _ = pipe.transformer(
        hidden_states=noisy_latent,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_prompt_embeds,
        return_dict=False,
    )[0]

    attn_cache = collect_attention_cache(pipe)
    if attn_cache is None:
        raise RuntimeError(
            "Attention cache is empty -- the attention processor did not record any data.\n"
            "This usually means you are using the standard pip-installed diffusers instead of\n"
            "Seg4Diff's vendored version which has custom attention caching.\n\n"
            "Fix: install the vendored diffusers from the Seg4Diff repo:\n"
            "  cd diffusers && pip install -e .\n"
        )
    return attn_cache


# ---------------------------------------------------------------------------
# Open-vocabulary segmentation (mirrors Seg4DiffOVSS.forward)
# ---------------------------------------------------------------------------

def segment_ovss(pipe, image_tensor, class_names, noise_steps, num_inference_steps,
                 attn_size=64, norm_before_merge=True, norm_after_merge=True, seed=42):
    """
    Open-vocabulary semantic segmentation using cross-attention token merging.
    Returns per-class attn_size x attn_size attention maps.
    """
    prompt = " ".join(class_names)
    num_tokens_per_class = [
        len(pipe.tokenizer(c)["input_ids"]) - 2 for c in class_names
    ]

    attn = extract_attention(pipe, image_tensor, prompt, noise_steps,
                             num_inference_steps, seed)
    slices = attn[0].mean(dim=0)

    if norm_before_merge:
        slices = slices - slices.min(dim=0, keepdim=True)[0]
        slices = slices / (slices.max(dim=0, keepdim=True)[0] + 1e-6)

    merged = torch.zeros(slices.shape[0], len(class_names), device=slices.device)
    st = 1  # skip <bos> token
    for i, n in enumerate(num_tokens_per_class):
        merged[:, i] = slices[:, st:st + n].mean(dim=-1)
        st += n

    if norm_after_merge:
        merged = merged - merged.min(dim=0, keepdim=True)[0]
        merged = merged / (merged.max(dim=0, keepdim=True)[0] + 1e-6)

    masks = merged.T.reshape(len(class_names), attn_size, attn_size)
    return masks


# ---------------------------------------------------------------------------
# Unsupervised segmentation (mirrors Seg4DiffUnsup.forward + mask_merge)
# ---------------------------------------------------------------------------

def symmetric_kl(x, y):
    quotient = torch.log(x) - torch.log(y)
    kl_1 = torch.sum(x * quotient, dim=(-2, -1)) / 2
    kl_2 = -torch.sum(y * quotient, dim=(-2, -1)) / 2
    return kl_1 + kl_2


def mask_merge(attns, kl_threshold):
    N, H, W = attns.shape
    flat = attns.reshape(N, -1)
    probs = torch.softmax(flat, dim=-1).view_as(attns)

    matched = set()
    new_list = []
    for i in range(N):
        if i in matched:
            continue
        matched.add(i)
        anchor = probs[i].unsqueeze(0).expand(N, -1, -1)
        kl_vals = symmetric_kl(anchor, probs)
        merge_mask = (kl_vals < kl_threshold).cpu()
        if merge_mask.sum() > 0:
            idxs = torch.nonzero(merge_mask.view(-1), as_tuple=False).squeeze().tolist()
            for idx in (idxs if isinstance(idxs, list) else [idxs]):
                matched.add(idx)
            new_list.append(probs[merge_mask].mean(dim=0))

    if new_list:
        return torch.stack(new_list, dim=0)
    return torch.empty((0, H, W), dtype=attns.dtype)


def att2mask(att, p=1.0):
    att = att - att.min(dim=-1, keepdim=True)[0]
    att = att / (att.max(dim=-1, keepdim=True)[0] + 1e-6)
    return att ** p


def segment_unsupervised(pipe, image_tensor, noise_steps, num_inference_steps,
                         attn_size=64, kl_threshold=0.1, output_power=2.0, seed=42):
    """
    Unsupervised segmentation via attention self-similarity and KL merging.
    """
    prompt = ""
    attn = extract_attention(pipe, image_tensor, prompt, noise_steps,
                             num_inference_steps, seed)
    slices = attn.mean(dim=1)
    slices = att2mask(slices, p=output_power)
    slices = rearrange(slices, "b (h w) l -> b l h w", h=attn_size, w=attn_size).squeeze(0)
    masks = mask_merge(slices, kl_threshold)
    return masks


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

COLORMAP = plt.colormaps["tab20"]


def colorize_segmentation(masks_np, alpha=0.6):
    """
    Given argmax mask (H, W) with integer labels, produce an RGBA overlay.
    """
    h, w = masks_np.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    unique = np.unique(masks_np)
    for idx in unique:
        color = np.array(COLORMAP(idx % 20))
        region = masks_np == idx
        overlay[region] = color
    overlay[..., 3] = alpha
    return (overlay * 255).astype(np.uint8)


def save_visualization(image_pil, masks, class_names, save_dir):
    """Save per-class mask images, heatmaps, argmax map, and overlay."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    orig_w, orig_h = image_pil.size
    img_np = np.array(image_pil.resize((512, 512)))
    n_classes = masks.shape[0]
    masks_np = masks.cpu().float().numpy()

    masks_up = F.interpolate(
        torch.tensor(masks_np).unsqueeze(0),
        size=(orig_h, orig_w), mode="bilinear", align_corners=False
    ).squeeze(0).numpy()

    # --- Per-class grayscale masks (original resolution) ---
    mask_dir = save_dir / "masks"
    mask_dir.mkdir(exist_ok=True)
    for i in range(n_classes):
        label = class_names[i] if class_names and i < len(class_names) else f"segment_{i}"
        binary = (masks_up[i] / (masks_up[i].max() + 1e-6) * 255).astype(np.uint8)
        Image.fromarray(binary, mode="L").save(mask_dir / f"{label}.png")

    # --- Argmax segmentation mask (original resolution) ---
    argmax_full = masks_up.argmax(axis=0).astype(np.uint8)

    n_unique = len(np.unique(argmax_full))
    if n_unique > 1:
        scaled = (argmax_full.astype(np.float32) / argmax_full.max() * 255).astype(np.uint8)
    else:
        scaled = argmax_full
    Image.fromarray(scaled, mode="L").save(save_dir / "argmax.png")

    # --- Color-coded argmax mask ---
    color_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    for idx in np.unique(argmax_full):
        color = (np.array(COLORMAP(idx % 20)[:3]) * 255).astype(np.uint8)
        color_mask[argmax_full == idx] = color
    Image.fromarray(color_mask).save(save_dir / "argmax_color.png")

    # --- Per-class heatmaps ---
    heatmap_dir = save_dir / "heatmaps"
    heatmap_dir.mkdir(exist_ok=True)
    for i in range(n_classes):
        label = class_names[i] if class_names and i < len(class_names) else f"segment_{i}"
        hmap = cv2.resize(masks_np[i], (512, 512))
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_np)
        ax[0].set_title("Image")
        ax[0].axis("off")
        im = ax[1].imshow(hmap, cmap="jet", vmin=0, vmax=1)
        ax[1].set_title(label)
        ax[1].axis("off")
        plt.colorbar(im, ax=ax[1], fraction=0.046)
        plt.tight_layout()
        fig.savefig(heatmap_dir / f"{label}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # --- Overlay + summary ---
    argmax_512 = cv2.resize(argmax_full, (512, 512), interpolation=cv2.INTER_NEAREST)
    overlay_rgba = colorize_segmentation(argmax_512, alpha=0.55)
    overlay_rgb = overlay_rgba[..., :3].astype(np.float32)
    overlay_a = overlay_rgba[..., 3:4].astype(np.float32) / 255.0
    blended = (img_np.astype(np.float32) * (1 - overlay_a) + overlay_rgb * overlay_a).astype(np.uint8)

    seg_color_512 = np.zeros((512, 512, 3), dtype=np.uint8)
    for idx in np.unique(argmax_512):
        seg_color_512[argmax_512 == idx] = (np.array(COLORMAP(idx % 20)[:3]) * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img_np)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(seg_color_512)
    axes[1].set_title("Segmentation Map")
    axes[1].axis("off")
    axes[2].imshow(blended)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    if class_names:
        used = np.unique(argmax_512)
        patches = []
        import matplotlib.patches as mpatches
        for idx in used:
            lbl = class_names[idx] if idx < len(class_names) else f"seg_{idx}"
            patches.append(mpatches.Patch(color=COLORMAP(idx % 20), label=lbl))
        axes[2].legend(handles=patches, loc="upper right", fontsize=8)

    plt.tight_layout()
    fig.savefig(save_dir / "summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    Image.fromarray(blended).save(save_dir / "overlay.png")
    print(f"  Saved results to {save_dir}")
    return argmax_full, blended


def save_comparison(image_pil, base_argmax, lora_argmax, base_blend, lora_blend,
                    class_names, save_path, caption=None):
    """Side-by-side comparison of base vs LoRA segmentation."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    img_np = np.array(image_pil.resize((512, 512)))

    def _argmax_to_rgb(argmax_map):
        h, w = argmax_map.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for idx in np.unique(argmax_map):
            rgb[argmax_map == idx] = (np.array(COLORMAP(idx % 20)[:3]) * 255).astype(np.uint8)
        return rgb

    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(_argmax_to_rgb(base_argmax))
    axes[0, 1].set_title("Base - Segmentation")
    axes[0, 1].axis("off")
    axes[0, 2].imshow(base_blend)
    axes[0, 2].set_title("Base - Overlay")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(img_np)
    axes[1, 0].set_title("Original Image")
    axes[1, 0].axis("off")
    axes[1, 1].imshow(_argmax_to_rgb(lora_argmax))
    axes[1, 1].set_title("LoRA - Segmentation")
    axes[1, 1].axis("off")
    axes[1, 2].imshow(lora_blend)
    axes[1, 2].set_title("LoRA - Overlay")
    axes[1, 2].axis("off")

    if class_names:
        import matplotlib.patches as mpatches
        used = set(np.unique(base_argmax)) | set(np.unique(lora_argmax))
        patches = [
            mpatches.Patch(color=COLORMAP(i % 20),
                           label=class_names[i] if i < len(class_names) else f"seg_{i}")
            for i in sorted(used)
        ]
        fig.legend(handles=patches, loc="lower center", ncol=min(len(patches), 8), fontsize=10)

    if caption:
        fig.suptitle(caption, fontsize=20, fontweight="bold", y=1.02)

    plt.tight_layout(rect=[0, 0.04, 1, 0.97 if caption else 1])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved comparison to {save_path}")


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image_tensor(image_path, resolution=1024, device="cuda"):
    """Load image, resize to resolution x resolution, normalise to [0,1], return (1,3,H,W) tensor."""
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize((resolution, resolution), Image.LANCZOS)
    tensor = torch.from_numpy(np.array(img_resized)).float().permute(2, 0, 1) / 255.0
    return img, tensor.unsqueeze(0).to(device)


def collect_image_paths(args):
    if args.image:
        return [args.image]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    folder = Path(args.image_dir)
    return sorted([str(p) for p in folder.iterdir() if p.suffix.lower() in exts])


def image_output_name(image_path):
    """Derive a unique, meaningful output folder name from an image path.

    For images from generate.py (e.g. outputs/generation/a_cat/lora/seed_42.png),
    uses the prompt slug + filename → "a_cat/seed_42".
    For standalone images (e.g. my_photo.jpg), just uses the stem → "my_photo".
    """
    p = Path(image_path)
    parent = p.parent.name          # e.g. "lora" or "base"
    grandparent = p.parent.parent.name  # e.g. "a_cat_sitting_on_a_windowsill"

    if parent in ("base", "lora") and grandparent:
        return Path(grandparent) / p.stem
    return Path(p.stem)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_segmentation(pipe, image_path, args, save_dir):
    """Run segmentation on a single image and save results."""
    image_pil, image_tensor = load_image_tensor(image_path, resolution=args.resolution, device=args.device)
    attn_size = args.resolution // 16  # VAE downscale (8) * patch embed (2)

    with torch.amp.autocast(args.device.split(":")[0], dtype=torch.float16 if args.fp16 else torch.float32):
        if args.unsupervised:
            masks = segment_unsupervised(
                pipe, image_tensor,
                noise_steps=args.noise_steps,
                num_inference_steps=args.num_inference_steps,
                attn_size=attn_size,
                kl_threshold=args.kl_threshold,
                output_power=args.output_power,
                seed=args.seed,
            )
            class_names = [f"segment_{i}" for i in range(masks.shape[0])]
        else:
            masks = segment_ovss(
                pipe, image_tensor, args.classes,
                noise_steps=args.noise_steps,
                num_inference_steps=args.num_inference_steps,
                attn_size=attn_size,
                norm_before_merge=args.norm_before_merge,
                norm_after_merge=args.norm_after_merge,
                seed=args.seed,
            )
            class_names = args.classes

    argmax_map, blended = save_visualization(image_pil, masks, class_names, save_dir)
    return masks, argmax_map, blended, class_names


def main():
    args = parse_args()

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    image_paths = collect_image_paths(args)
    print(f"Found {len(image_paths)} image(s) to segment.")
    mode = "unsupervised" if args.unsupervised else f"open-vocab ({', '.join(args.classes)})"
    print(f"Segmentation mode: {mode}")
    print(f"LoRA weights: {args.lora_weights or 'None (base model only)'}")
    print("-" * 60)

    use_lora = args.lora_weights is not None
    lora_targets = None
    if args.lora_targets:
        lora_targets = [m.strip() for m in args.lora_targets.split(",")]

    # --- Phase 1: Base model ---
    pipe = load_pipeline(args.model_id, fp16=args.fp16, device=args.device)
    setup_attention_caching(pipe, args.attention_layers)

    base_results = {}
    for img_path in image_paths:
        out_name = image_output_name(img_path)
        print(f"\n[Base] Segmenting: {out_name}")
        img_dir = root / out_name / "base"
        masks, argmax, blend, cnames = run_segmentation(
            pipe, img_path, args, img_dir
        )
        base_results[img_path] = (masks, argmax, blend, cnames)

    # --- Phase 2: LoRA model (if provided) ---
    if use_lora:
        del pipe
        torch.cuda.empty_cache()

        pipe = load_pipeline(args.model_id, fp16=args.fp16, device=args.device)
        setup_attention_caching(pipe, args.attention_layers)
        pipe = apply_lora(
            pipe, args.lora_weights,
            lora_rank=args.lora_rank,
            target_modules=lora_targets,
            lora_layers=args.lora_layers,
            weight_prefix=args.lora_weight_prefix,
        )

        for img_path in image_paths:
            out_name = image_output_name(img_path)
            print(f"\n[LoRA] Segmenting: {out_name}")
            lora_dir = root / out_name / "lora"
            lora_masks, lora_argmax, lora_blend, cnames = run_segmentation(
                pipe, img_path, args, lora_dir
            )

            image_pil = Image.open(img_path).convert("RGB")
            base_masks, base_argmax, base_blend, _ = base_results[img_path]

            prompt_slug = Path(img_path).parent.parent.name if Path(img_path).parent.name in ("base", "lora") else Path(img_path).stem
            caption = prompt_slug.replace("_", " ")

            save_comparison(
                image_pil, base_argmax, lora_argmax, base_blend, lora_blend,
                cnames, root / out_name / "comparison.png",
                caption=caption,
            )

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Segmentation complete!")
    print(f"Results saved to: {root}")
    for img_path in image_paths:
        out_name = image_output_name(img_path)
        print(f"\n  {out_name}/")
        print(f"    base/")
        print(f"      masks/        : Per-class grayscale masks")
        print(f"      heatmaps/     : Per-class attention heatmaps")
        print(f"      argmax.png    : Grayscale segmentation mask")
        print(f"      argmax_color.png : Color-coded segmentation")
        print(f"      overlay.png   : Blended overlay on original")
        print(f"      summary.png   : 3-panel overview")
        if use_lora:
            print(f"    lora/           : Same structure as base/")
            print(f"    comparison.png  : Base vs LoRA side-by-side")
    print("=" * 60)


if __name__ == "__main__":
    main()
