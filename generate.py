"""
SD3 Image Generation Script (based on Seg4Diff framework)

Generates images using Stable Diffusion 3 with optional LoRA weight loading.
When LoRA weights are provided, generates side-by-side comparisons showing
the difference between base model and LoRA-finetuned outputs.

Usage:
    # Basic generation
    python generate.py --prompt "a cat sitting on a windowsill"

    # Multiple prompts
    python generate.py --prompt "a cat" "a dog in the park"

    # With LoRA weights (generates both base and LoRA versions for comparison)
    python generate.py --prompt "a cat" --lora_weights lora_weights/lora_weights.pth

    # Full options
    python generate.py \
        --prompt "a photorealistic landscape" \
        --output_dir ./outputs/generation \
        --num_images 3 \
        --guidance_scale 7.5 \
        --num_inference_steps 28 \
        --seed 42 \
        --resolution 1024 \
        --lora_weights lora_weights/lora_weights.pth

Output structure:
    outputs/generation/
    └── a_cat_sitting_on_a_windowsill/
        ├── base/
        │   ├── seed_42.png
        │   └── seed_43.png
        ├── lora/
        │   ├── seed_42.png
        │   └── seed_43.png
        └── comparison.png
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from diffusers import StableDiffusion3Pipeline
from peft import LoraConfig, set_peft_model_state_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate images with Stable Diffusion 3, optionally with LoRA weights"
    )
    parser.add_argument(
        "--prompt", type=str, nargs="+", required=True,
        help="One or more text prompts for image generation",
    )
    parser.add_argument(
        "--negative_prompt", type=str, default=None,
        help="Negative prompt to guide generation away from",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs/generation",
        help="Root directory for generated images (default: ./outputs/generation)",
    )
    parser.add_argument(
        "--num_images", type=int, default=1,
        help="Number of images to generate per prompt (default: 1)",
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=7.5,
        help="Classifier-free guidance scale (default: 7.5)",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=28,
        help="Number of denoising steps (default: 28)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--resolution", type=int, default=1024,
        help="Output image resolution (default: 1024)",
    )
    parser.add_argument(
        "--model_id", type=str,
        default="stabilityai/stable-diffusion-3-medium-diffusers",
        help="HuggingFace model ID for SD3",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Torch device (default: cuda). Use cuda:1 for second GPU.",
    )
    parser.add_argument(
        "--fp16", action="store_true", default=True,
        help="Use float16 precision (default: True)",
    )
    parser.add_argument(
        "--no_fp16", action="store_true",
        help="Disable float16 precision",
    )

    # LoRA arguments
    parser.add_argument(
        "--lora_weights", type=str, default=None,
        help="Path to LoRA weights file (.pth). When provided, generates both "
             "base and LoRA images for comparison.",
    )
    parser.add_argument(
        "--lora_rank", type=int, default=16,
        help="LoRA rank (default: 16, matching Seg4Diff config)",
    )
    parser.add_argument(
        "--lora_targets", type=str, default=None,
        help="Comma-separated list of LoRA target modules "
             "(default: attn.to_k,attn.to_q,attn.to_v,attn.to_out.0)",
    )
    parser.add_argument(
        "--lora_layers", type=int, nargs="*", default=None,
        help="Specific transformer block indices to apply LoRA to "
             "(default: all blocks)",
    )
    parser.add_argument(
        "--lora_weight_prefix", type=str, default="backbone.transformer.",
        help="Prefix to strip from LoRA state dict keys "
             "(default: backbone.transformer.)",
    )

    args = parser.parse_args()
    if args.no_fp16:
        args.fp16 = False
    return args


def load_pipeline(model_id, fp16=True, device="cuda"):
    """Load the SD3 pipeline following Seg4Diff's pattern."""
    print(f"Loading SD3 pipeline from: {model_id}")
    dtype = torch.float16 if fp16 else None
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        text_encoder_3=None,
        torch_dtype=dtype,
    )
    pipe.to(device)
    print("Pipeline loaded successfully.")
    return pipe


def apply_lora(pipe, lora_weights_path, lora_rank=16, target_modules=None,
               lora_layers=None, weight_prefix="backbone.transformer."):
    """
    Apply LoRA adapter to the transformer, following the Seg4Diff backbone pattern
    (see sd3_backbone.py lines 57-110).
    """
    if target_modules is None:
        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
        ]

    if lora_layers is not None:
        target_modules = [
            f"transformer_blocks.{block}.{module}"
            for block in lora_layers
            for module in target_modules
        ]

    transformer_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    pipe.transformer.add_adapter(transformer_lora_config)

    for name, param in pipe.transformer.named_parameters():
        if "lora" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    def preprocess_state_dict(state_dict, prefix_to_remove):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(prefix_to_remove):
                k = k[len(prefix_to_remove):]
            k = k.replace(".default.", ".")
            new_state_dict[k] = v
        return new_state_dict

    print(f"Loading LoRA weights from: {lora_weights_path}")
    try:
        state_dict = torch.load(lora_weights_path, map_location="cpu", weights_only=False)
    except Exception:
        from safetensors.torch import load_file
        state_dict = load_file(lora_weights_path)

    if "model" in state_dict:
        state_dict_to_load = preprocess_state_dict(state_dict["model"], weight_prefix)
    else:
        state_dict_to_load = preprocess_state_dict(state_dict, weight_prefix)

    outcome = set_peft_model_state_dict(pipe.transformer, state_dict_to_load)
    print(f"LoRA weights loaded. Unexpected keys: {outcome.unexpected_keys[:5] if outcome.unexpected_keys else 'none'}")

    return pipe


def slugify(text, max_len=60):
    """Convert a prompt string into a filesystem-safe folder name."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_-]+", "_", slug)
    return slug[:max_len].rstrip("_")


def generate_images(pipe, prompt, negative_prompt=None, num_images=1,
                    guidance_scale=7.5, num_inference_steps=28,
                    seed=42, resolution=1024):
    """Generate images for a single prompt."""
    images = []
    for i in range(num_images):
        generator = torch.manual_seed(seed + i)
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_inference_steps,
            height=resolution,
            width=resolution,
        )
        images.append(result.images[0])
    return images


def create_comparison_grid(base_images, lora_images, prompt, seed):
    """Create a side-by-side comparison image (base vs LoRA)."""
    n = len(base_images)
    img_w, img_h = base_images[0].size
    label_height = 40
    padding = 10
    prompt_height = 50

    grid_w = (img_w * 2) + (padding * 3)
    grid_h = prompt_height + (img_h + label_height + padding) * n + padding

    grid = Image.new("RGB", (grid_w, grid_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
        font_small = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
        font_small = font

    truncated = prompt if len(prompt) < 80 else prompt[:77] + "..."
    draw.text((padding, 10), f'Prompt: "{truncated}"', fill="white", font=font_small)

    for i in range(n):
        y_offset = prompt_height + i * (img_h + label_height + padding)

        draw.text((padding, y_offset), f"Base (seed={seed + i})",
                  fill=(100, 200, 255), font=font)
        grid.paste(base_images[i], (padding, y_offset + label_height))

        draw.text((img_w + padding * 2, y_offset), f"LoRA (seed={seed + i})",
                  fill=(255, 180, 100), font=font)
        grid.paste(lora_images[i], (img_w + padding * 2, y_offset + label_height))

    return grid


def main():
    args = parse_args()

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {root}")
    print(f"Prompts: {args.prompt}")
    print(f"LoRA weights: {args.lora_weights or 'None (base model only)'}")
    print("-" * 60)

    use_lora = args.lora_weights is not None
    lora_target_modules = None
    if args.lora_targets:
        lora_target_modules = [m.strip() for m in args.lora_targets.split(",")]

    # --- Phase 1: Generate base images ---
    pipe = load_pipeline(args.model_id, fp16=args.fp16, device=args.device)

    all_base_images = {}
    for pi, prompt in enumerate(args.prompt):
        print(f"\n[Base] Generating {args.num_images} image(s) for prompt {pi+1}/{len(args.prompt)}: \"{prompt}\"")
        images = generate_images(
            pipe, prompt,
            negative_prompt=args.negative_prompt,
            num_images=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            resolution=args.resolution,
        )
        all_base_images[pi] = images

        prompt_dir = root / slugify(prompt)
        base_dir = prompt_dir / "base"
        base_dir.mkdir(parents=True, exist_ok=True)
        for idx, img in enumerate(images):
            save_path = base_dir / f"seed_{args.seed + idx}.png"
            img.save(save_path)
            print(f"  Saved: {save_path}")

    # --- Phase 2: Generate LoRA images (if weights provided) ---
    if use_lora:
        del pipe
        torch.cuda.empty_cache()

        pipe = load_pipeline(args.model_id, fp16=args.fp16, device=args.device)
        pipe = apply_lora(
            pipe, args.lora_weights,
            lora_rank=args.lora_rank,
            target_modules=lora_target_modules,
            lora_layers=args.lora_layers,
            weight_prefix=args.lora_weight_prefix,
        )

        for pi, prompt in enumerate(args.prompt):
            print(f"\n[LoRA] Generating {args.num_images} image(s) for prompt {pi+1}/{len(args.prompt)}: \"{prompt}\"")
            lora_images = generate_images(
                pipe, prompt,
                negative_prompt=args.negative_prompt,
                num_images=args.num_images,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                seed=args.seed,
                resolution=args.resolution,
            )

            prompt_dir = root / slugify(prompt)
            lora_dir = prompt_dir / "lora"
            lora_dir.mkdir(parents=True, exist_ok=True)
            for idx, img in enumerate(lora_images):
                save_path = lora_dir / f"seed_{args.seed + idx}.png"
                img.save(save_path)
                print(f"  Saved: {save_path}")

            comparison = create_comparison_grid(
                all_base_images[pi], lora_images, prompt, args.seed
            )
            comp_path = prompt_dir / "comparison.png"
            comparison.save(comp_path)
            print(f"  Saved comparison: {comp_path}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Generation complete!")
    print(f"Results saved to: {root}")
    for pi, prompt in enumerate(args.prompt):
        slug = slugify(prompt)
        print(f"\n  {slug}/")
        print(f"    base/seed_*.png   : Base SD3 images")
        if use_lora:
            print(f"    lora/seed_*.png   : LoRA-finetuned images")
            print(f"    comparison.png    : Side-by-side grid")
    print("=" * 60)


if __name__ == "__main__":
    main()
