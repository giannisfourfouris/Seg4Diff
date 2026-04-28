"""
Batch Generation + Segmentation Pipeline (COCO metadata-driven)

Reads a metadata.json file and for each entry:
1. Generates an image from the first caption (base SD3 + LoRA SD3)
2. Segments each generated image using the entry's object classes
3. Creates a comparison figure with the caption as title

Usage:
    python batch_pipeline.py \
        --metadata path/to/metadata.json \
        --lora_weights lora_weights/sa1b/lora_weights.pth

    # Process only 10 images starting from index 20
    python batch_pipeline.py \
        --metadata path/to/metadata.json \
        --lora_weights lora_weights/sa1b/lora_weights.pth \
        --limit 10 --start 20

    # Resume: skip images that already have output
    python batch_pipeline.py \
        --metadata path/to/metadata.json \
        --lora_weights lora_weights/sa1b/lora_weights.pth \
        --skip_existing

Output structure:
    outputs/batch/
    ├── manifest.json
    ├── 000000000139/
    │   ├── base/
    │   │   ├── generated.png
    │   │   ├── argmax.png
    │   │   ├── argmax_color.png
    │   │   └── overlay.png
    │   ├── lora/
    │   │   ├── generated.png
    │   │   ├── argmax.png
    │   │   ├── argmax_color.png
    │   │   └── overlay.png
    │   └── comparison.png
    └── ...
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from segment import (
    COLORMAP,
    apply_lora,
    colorize_segmentation,
    load_pipeline,
    map_to_coco_ids,
    segment_ovss,
    segment_unsupervised,
    setup_attention_caching,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch generation + segmentation from COCO metadata"
    )

    # Input
    parser.add_argument(
        "--metadata", type=str, required=True,
        help="Path to metadata.json (list of {image, captions, classes, ...})",
    )
    parser.add_argument(
        "--caption_index", type=int, default=0,
        help="Which caption to use per image (default: 0 = first)",
    )

    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs/batch")

    # Subset control
    parser.add_argument("--limit", type=int, default=None, help="Max images to process")
    parser.add_argument("--start", type=int, default=0, help="Start index (for resuming)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip if output already exists")

    # Model
    parser.add_argument(
        "--model_id", type=str,
        default="stabilityai/stable-diffusion-3-medium-diffusers",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no_fp16", action="store_true")

    # Generation
    parser.add_argument("--negative_prompt", type=str, default=None,
                        help="Negative prompt for generation")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=1024)

    # Segmentation
    parser.add_argument("--noise_steps", type=int, default=8)
    parser.add_argument("--attention_layers", type=int, nargs="*", default=[9])
    parser.add_argument("--norm_before_merge", action="store_true", default=False)
    parser.add_argument("--norm_after_merge", action="store_true", default=False)
    parser.add_argument("--kl_threshold", type=float, default=0.035,
                        help="KL divergence threshold for unsupervised mask merging (default: 0.035)")
    parser.add_argument("--output_power", type=float, default=1.0,
                        help="Power exponent for att2mask in unsupervised mode (default: 1.0)")

    # LoRA
    parser.add_argument("--lora_weights", type=str, default=None)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_targets", type=str, default=None)
    parser.add_argument("--lora_layers", type=int, nargs="*", default=None)
    parser.add_argument("--lora_weight_prefix", type=str, default="backbone.transformer.")

    args = parser.parse_args()
    if args.no_fp16:
        args.fp16 = False
    return args


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_image(pipe, prompt, negative_prompt=None, guidance_scale=7.5,
                   num_inference_steps=28, seed=42, resolution=1024):
    generator = torch.manual_seed(seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        generator=generator,
        num_inference_steps=num_inference_steps,
        height=resolution,
        width=resolution,
    )
    return result.images[0]


# ---------------------------------------------------------------------------
# Attention caching toggle
# ---------------------------------------------------------------------------

def disable_attention_caching(pipe):
    """Turn off attention recording so generation doesn't accumulate maps."""
    for blk in pipe.transformer.transformer_blocks:
        blk.attn.processor.attn_cache = None


def enable_attention_caching(pipe, attention_layers):
    """Re-enable attention recording on selected layers."""
    for l, blk in enumerate(pipe.transformer.transformer_blocks):
        if l in attention_layers:
            blk.attn.processor.attn_cache = []


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def pil_to_tensor(image_pil, resolution, device):
    img = image_pil.convert("RGB").resize((resolution, resolution), Image.LANCZOS)
    tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    return tensor.unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Save helpers (slimmed-down: only argmax, argmax_color, overlay)
# ---------------------------------------------------------------------------

def save_segmentation(image_pil, masks, class_names, save_dir):
    """Save argmax.png, argmax_color.png, overlay.png. Returns (argmax_full, blended_512)."""
    save_dir = Path(save_dir)

    orig_w, orig_h = image_pil.size
    img_512 = np.array(image_pil.resize((512, 512)))
    masks_np = masks.cpu().float().numpy()

    masks_up = F.interpolate(
        torch.tensor(masks_np).unsqueeze(0),
        size=(orig_h, orig_w), mode="bilinear", align_corners=False,
    ).squeeze(0).numpy()

    argmax_full = masks_up.argmax(axis=0).astype(np.uint8)

    # Raw class-index argmax (values 0, 1, 2, ...)
    Image.fromarray(argmax_full, mode="L").save(save_dir / "argmax.png")

    # RNS-compatible mask with fixed COCO class IDs
    coco_mask = map_to_coco_ids(argmax_full, class_names)
    Image.fromarray(coco_mask, mode="L").save(save_dir / "argmax_rns.png")

    # Color-coded argmax
    color_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    for idx in np.unique(argmax_full):
        color = (np.array(COLORMAP(idx % 20)[:3]) * 255).astype(np.uint8)
        color_mask[argmax_full == idx] = color
    Image.fromarray(color_mask).save(save_dir / "argmax_color.png")

    # Overlay at 512x512
    argmax_512 = cv2.resize(argmax_full, (512, 512), interpolation=cv2.INTER_NEAREST)
    overlay_rgba = colorize_segmentation(argmax_512, alpha=0.55)
    overlay_rgb = overlay_rgba[..., :3].astype(np.float32)
    overlay_a = overlay_rgba[..., 3:4].astype(np.float32) / 255.0
    blended = (img_512.astype(np.float32) * (1 - overlay_a) + overlay_rgb * overlay_a).astype(np.uint8)
    Image.fromarray(blended).save(save_dir / "overlay.png")

    return argmax_full, blended


def _argmax_to_rgb(argmax_map):
    h, w = argmax_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx in np.unique(argmax_map):
        rgb[argmax_map == idx] = (np.array(COLORMAP(idx % 20)[:3]) * 255).astype(np.uint8)
    return rgb


def save_comparison(base_gen_pil, lora_gen_pil, base_argmax, lora_argmax,
                    base_blend, lora_blend, class_names, caption, save_path):
    """Side-by-side: base generated/seg/overlay vs LoRA generated/seg/overlay."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))

    base_img = np.array(base_gen_pil.resize((512, 512)))
    lora_img = np.array(lora_gen_pil.resize((512, 512)))

    base_seg = _argmax_to_rgb(cv2.resize(base_argmax, (512, 512), interpolation=cv2.INTER_NEAREST))
    lora_seg = _argmax_to_rgb(cv2.resize(lora_argmax, (512, 512), interpolation=cv2.INTER_NEAREST))

    axes[0, 0].imshow(base_img)
    axes[0, 0].set_title("Base - Generated")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(base_seg)
    axes[0, 1].set_title("Base - Segmentation")
    axes[0, 1].axis("off")
    axes[0, 2].imshow(base_blend)
    axes[0, 2].set_title("Base - Overlay")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(lora_img)
    axes[1, 0].set_title("LoRA - Generated")
    axes[1, 0].axis("off")
    axes[1, 1].imshow(lora_seg)
    axes[1, 1].set_title("LoRA - Segmentation")
    axes[1, 1].axis("off")
    axes[1, 2].imshow(lora_blend)
    axes[1, 2].set_title("LoRA - Overlay")
    axes[1, 2].axis("off")

    if class_names:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    with open(args.metadata) as f:
        metadata = json.load(f)

    metadata = metadata[args.start:]
    if args.limit:
        metadata = metadata[:args.limit]

    entries = []
    for entry in metadata:
        img_id = Path(entry["image"]).stem
        caption = entry["captions"][args.caption_index]
        classes = entry["classes"]
        if "background" not in [c.lower() for c in classes]:
            classes = classes + ["background"]
        entries.append({"id": img_id, "caption": caption, "classes": classes})

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    manifest = {e["id"]: {"caption": e["caption"], "classes": e["classes"]} for e in entries}
    with open(root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    use_lora = args.lora_weights is not None
    lora_targets = [m.strip() for m in args.lora_targets.split(",")] if args.lora_targets else None
    attn_size = args.resolution // 16

    print(f"Processing {len(entries)} images")
    print(f"Output: {root}")
    print(f"LoRA: {args.lora_weights or 'None (base only)'}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Phase 1: Base model — generate + segment
    # ------------------------------------------------------------------
    pipe = load_pipeline(args.model_id, fp16=args.fp16, device=args.device)
    setup_attention_caching(pipe, args.attention_layers)

    t0 = time.time()
    for i, entry in enumerate(entries):
        img_id, caption, classes = entry["id"], entry["caption"], entry["classes"]
        out_dir = root / img_id / "base"

        if args.skip_existing and (out_dir / "overlay.png").exists():
            print(f"  [{i+1}/{len(entries)}] Skip {img_id} (exists)")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[Base] ({i+1}/{len(entries)}) {img_id}: {caption[:70]}")

        # Generate
        disable_attention_caching(pipe)
        gen_img = generate_image(
            pipe, caption,
            negative_prompt=args.negative_prompt,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            resolution=args.resolution,
        )
        gen_img.save(out_dir / "generated.png")

        # Segment
        enable_attention_caching(pipe, args.attention_layers)
        img_tensor = pil_to_tensor(gen_img, args.resolution, args.device)
        with torch.amp.autocast(args.device.split(":")[0],
                                dtype=torch.float16 if args.fp16 else torch.float32):
            if classes:
                masks = segment_ovss(
                    pipe, img_tensor, classes,
                    noise_steps=args.noise_steps,
                    num_inference_steps=args.num_inference_steps,
                    attn_size=attn_size,
                    norm_before_merge=args.norm_before_merge,
                    norm_after_merge=args.norm_after_merge,
                    seed=args.seed,
                )
            else:
                masks = segment_unsupervised(
                    pipe, img_tensor,
                    noise_steps=args.noise_steps,
                    num_inference_steps=args.num_inference_steps,
                    attn_size=attn_size,
                    kl_threshold=args.kl_threshold,
                    output_power=args.output_power,
                    seed=args.seed,
                )
                classes = [f"segment_{i}" for i in range(masks.shape[0])]
                entry["classes"] = classes

        save_segmentation(gen_img, masks, classes, out_dir)
        elapsed = time.time() - t0
        avg = elapsed / (i + 1)
        remaining = avg * (len(entries) - i - 1)
        print(f"  Done ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    # ------------------------------------------------------------------
    # Phase 2: LoRA model — generate + segment + comparison
    # ------------------------------------------------------------------
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

        t0 = time.time()
        for i, entry in enumerate(entries):
            img_id, caption, classes = entry["id"], entry["caption"], entry["classes"]
            out_dir = root / img_id / "lora"
            comp_path = root / img_id / "comparison.png"

            if args.skip_existing and comp_path.exists():
                print(f"  [{i+1}/{len(entries)}] Skip {img_id} (exists)")
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n[LoRA] ({i+1}/{len(entries)}) {img_id}: {caption[:70]}")

            # Generate
            disable_attention_caching(pipe)
            gen_img = generate_image(
                pipe, caption,
                negative_prompt=args.negative_prompt,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                seed=args.seed,
                resolution=args.resolution,
            )
            gen_img.save(out_dir / "generated.png")

            # Segment
            enable_attention_caching(pipe, args.attention_layers)
            img_tensor = pil_to_tensor(gen_img, args.resolution, args.device)
            with torch.amp.autocast(args.device.split(":")[0],
                                    dtype=torch.float16 if args.fp16 else torch.float32):
                if classes:
                    masks = segment_ovss(
                        pipe, img_tensor, classes,
                        noise_steps=args.noise_steps,
                        num_inference_steps=args.num_inference_steps,
                        attn_size=attn_size,
                        norm_before_merge=args.norm_before_merge,
                        norm_after_merge=args.norm_after_merge,
                        seed=args.seed,
                    )
                else:
                    masks = segment_unsupervised(
                        pipe, img_tensor,
                        noise_steps=args.noise_steps,
                        num_inference_steps=args.num_inference_steps,
                        attn_size=attn_size,
                        kl_threshold=args.kl_threshold,
                        output_power=args.output_power,
                        seed=args.seed,
                    )
                    classes = [f"segment_{i}" for i in range(masks.shape[0])]

            lora_argmax, lora_blend = save_segmentation(gen_img, masks, classes, out_dir)

            # Comparison — reload base results from disk
            base_dir = root / img_id / "base"
            base_gen_pil = Image.open(base_dir / "generated.png")
            base_argmax = np.array(Image.open(base_dir / "argmax.png"))
            base_blend = np.array(Image.open(base_dir / "overlay.png"))

            save_comparison(
                base_gen_pil, gen_img, base_argmax, lora_argmax,
                base_blend, lora_blend, classes, caption, comp_path,
            )

            elapsed = time.time() - t0
            avg = elapsed / (i + 1)
            remaining = avg * (len(entries) - i - 1)
            print(f"  Done ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Batch pipeline complete!")
    print(f"Results: {root}")
    print(f"Manifest: {root / 'manifest.json'}")
    print(f"Images processed: {len(entries)}")
    if use_lora:
        print("Each folder contains: base/ + lora/ + comparison.png")
    else:
        print("Each folder contains: base/")
    print("=" * 60)


if __name__ == "__main__":
    main()
