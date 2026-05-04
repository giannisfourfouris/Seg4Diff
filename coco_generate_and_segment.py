"""
Batch Generation + Segmentation Pipeline (COCO metadata-driven)

Reads seg4diff_coco/info/metadata.json and for each entry:
1. Generates an image from the selected caption using SD3 (base or LoRA)
2. Segments the generated image using the entry's COCO object classes
3. Saves results in RnS-compatible folder structure

The mode suffix (base/lora) is determined by whether --lora_weights is provided.
To get both, run the script twice (once without LoRA, once with).

Usage:
    # Base model only
    python coco_generate_and_segment.py \
        --metadata seg4diff_coco/info/metadata.json \
        --output_dir seg4diff_coco

    # LoRA model only
    python coco_generate_and_segment.py \
        --metadata seg4diff_coco/info/metadata.json \
        --output_dir seg4diff_coco \
        --lora_weights lora_weights/sa1b/lora_weights.pth

    # Process only 10 images starting from index 20
    python coco_generate_and_segment.py \
        --metadata seg4diff_coco/info/metadata.json \
        --limit 10 --start 20

    # Resume: skip images that already have output
    python coco_generate_and_segment.py \
        --metadata seg4diff_coco/info/metadata.json \
        --skip_existing

    # Multi-GPU: split work across 4 GPUs automatically
    python coco_generate_and_segment.py \
        --metadata seg4diff_coco/info/metadata.json \
        --num_gpus 4

Output structure (RnS-compatible, split by train2017/val2017, per caption index):
    seg4diff_coco/
    ├── generated_images_base_c0/train2017/000000000139.jpg
    ├── generated_images_base_c0/val2017/000000000139.jpg
    ├── generated_masks_base_c0/train2017/000000000139_instanceTrainIds.png
    ├── generated_masks_base_c0/val2017/000000000139_instanceTrainIds.png
    ├── generated_images_base_c1/train2017/...
    └── generated_masks_lora_c0/train2017/...
"""

import argparse
import json
import logging
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from segment import (
    apply_lora,
    load_pipeline,
    segment_ovss,
    segment_unsupervised,
    setup_attention_caching,
)

log = logging.getLogger("batch_pipeline")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: Path, suffix: str, device: str):
    """Configure logging to console + timestamped file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"coco_gen_seg_{suffix}_{device.replace(':', '')}_{timestamp}.log"

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)

    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.addHandler(file_handler)
    log.addHandler(console_handler)

    log.info("Log file: %s", log_file)
    return log_file


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
    parser.add_argument("--output_dir", type=str, default="./seg4diff_coco")

    # Subset / parallelism
    parser.add_argument("--limit", type=int, default=None, help="Max images to process")
    parser.add_argument("--start", type=int, default=0, help="Start index (for resuming)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip if output already exists")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs for parallel processing (default: 1)")

    # Sampling
    parser.add_argument("--train_pct", type=float, default=None,
                        help="Random percentage of train images to keep (e.g., 5 means 5%%)")
    parser.add_argument("--val_pct", type=float, default=None,
                        help="Random percentage of val images to keep (e.g., 4 means 4%%)")
    parser.add_argument("--sample_seed", type=int, default=42,
                        help="Random seed for reproducible sampling (default: 42)")

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
    parser.add_argument("--resolution", type=int, default=512)

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

    uses_sampling = args.train_pct is not None or args.val_pct is not None
    uses_slicing = args.start != 0 or args.limit is not None
    if uses_sampling and uses_slicing:
        parser.error("--train_pct/--val_pct and --start/--limit are mutually exclusive")

    return args


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_image(pipe, prompt, negative_prompt=None, guidance_scale=7.5,
                   num_inference_steps=28, seed=42, resolution=512):
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
# COCO class-to-ID mapping (background=0, 80 COCO object classes=1..80)
# ---------------------------------------------------------------------------

COCO_CLASSES = (
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush',
)

COCO_NAME_TO_ID = {name: idx for idx, name in enumerate(COCO_CLASSES)}


def map_to_coco_ids(argmax_local, class_names):
    """Remap local argmax indices (0..N-1) to fixed COCO class IDs."""
    h, w = argmax_local.shape
    coco_mask = np.zeros((h, w), dtype=np.uint8)
    for local_idx, name in enumerate(class_names):
        name_lower = name.lower().strip()
        coco_id = COCO_NAME_TO_ID.get(name_lower)
        if coco_id is None:
            log.warning("'%s' not in COCO classes -> 255 (ignore)", name)
            coco_id = 255
        coco_mask[argmax_local == local_idx] = coco_id
    return coco_mask


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_mask(image_pil, masks, class_names, mask_path):
    """Produce argmax mask with fixed COCO class IDs and save to mask_path."""
    orig_w, orig_h = image_pil.size
    masks_np = masks.cpu().float().numpy()

    masks_up = F.interpolate(
        torch.tensor(masks_np).unsqueeze(0),
        size=(orig_h, orig_w), mode="bilinear", align_corners=False,
    ).squeeze(0).numpy()

    argmax_full = masks_up.argmax(axis=0).astype(np.uint8)
    coco_mask = map_to_coco_ids(argmax_full, class_names)
    Image.fromarray(coco_mask, mode="L").save(mask_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _launch_multi_gpu(args, total_entries):
    """Split work across GPUs by launching parallel subprocesses."""
    chunk = (total_entries + args.num_gpus - 1) // args.num_gpus
    procs = []
    for gpu_idx in range(args.num_gpus):
        worker_start = args.start + gpu_idx * chunk
        worker_limit = min(chunk, total_entries - gpu_idx * chunk)
        if worker_limit <= 0:
            break

        cmd = [
            sys.executable, __file__,
            "--metadata", args.metadata,
            "--output_dir", args.output_dir,
            "--caption_index", str(args.caption_index),
            "--start", str(worker_start),
            "--limit", str(worker_limit),
            "--device", f"cuda:{gpu_idx}",
            "--model_id", args.model_id,
            "--guidance_scale", str(args.guidance_scale),
            "--num_inference_steps", str(args.num_inference_steps),
            "--seed", str(args.seed),
            "--resolution", str(args.resolution),
            "--noise_steps", str(args.noise_steps),
            "--kl_threshold", str(args.kl_threshold),
            "--output_power", str(args.output_power),
            "--lora_rank", str(args.lora_rank),
            "--lora_weight_prefix", args.lora_weight_prefix,
            "--num_gpus", "1",
        ]
        if args.skip_existing:
            cmd.append("--skip_existing")
        if args.fp16:
            cmd.append("--fp16")
        else:
            cmd.append("--no_fp16")
        if args.norm_before_merge:
            cmd.append("--norm_before_merge")
        if args.norm_after_merge:
            cmd.append("--norm_after_merge")
        if args.negative_prompt:
            cmd.extend(["--negative_prompt", args.negative_prompt])
        if args.lora_weights:
            cmd.extend(["--lora_weights", args.lora_weights])
        if args.lora_targets:
            cmd.extend(["--lora_targets", args.lora_targets])
        if args.lora_layers:
            cmd.extend(["--lora_layers"] + [str(l) for l in args.lora_layers])
        if args.attention_layers:
            cmd.extend(["--attention_layers"] + [str(l) for l in args.attention_layers])

        log.info("  GPU %d: images [%d..%d)", gpu_idx, worker_start, worker_start + worker_limit)
        procs.append(subprocess.Popen(cmd))

    log.info("Waiting for %d workers ...", len(procs))
    failed = []
    for gpu_idx, proc in enumerate(procs):
        rc = proc.wait()
        if rc != 0:
            failed.append(gpu_idx)
    return failed


def _stratified_sample(entries, pct, rng, caption_index=0):
    """Sample pct% of entries with class-coverage-first strategy.

    Uses caption_classes[caption_index] for class coverage (what will actually
    be segmented), falling back to GT classes if caption_classes is missing.

    1. Greedily pick images to cover as many classes as possible.
    2. Fill remaining budget with random samples from leftovers.
    """
    k = max(1, int(len(entries) * pct / 100))
    if k >= len(entries):
        return list(entries)

    def _get_classes(e):
        cc = e.get("caption_classes")
        if cc and caption_index < len(cc):
            return cc[caption_index].get("classes", [])
        return e.get("classes", [])

    uncovered = set()
    class_to_entries = {}
    for idx, e in enumerate(entries):
        for cls in _get_classes(e):
            uncovered.add(cls)
            class_to_entries.setdefault(cls, []).append(idx)

    selected_indices = set()

    # Greedy: pick image that covers the most uncovered classes
    while uncovered and len(selected_indices) < k:
        best_idx, best_gain = None, 0
        candidates = set()
        for cls in uncovered:
            candidates.update(class_to_entries.get(cls, []))
        for idx in candidates:
            if idx in selected_indices:
                continue
            gain = sum(1 for cls in _get_classes(entries[idx]) if cls in uncovered)
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx is None:
            break
        selected_indices.add(best_idx)
        for cls in _get_classes(entries[best_idx]):
            uncovered.discard(cls)

    # Fill remaining budget randomly
    remaining_indices = [i for i in range(len(entries)) if i not in selected_indices]
    fill_count = k - len(selected_indices)
    if fill_count > 0 and remaining_indices:
        selected_indices.update(rng.sample(remaining_indices, min(fill_count, len(remaining_indices))))

    coverage_count = len(class_to_entries) - len(uncovered)
    log.info("  Stratified sample: %d/%d images, %d/%d classes covered (caption_index=%d)",
             len(selected_indices), len(entries), coverage_count, len(class_to_entries),
             caption_index)

    return [entries[i] for i in sorted(selected_indices)]


def main():
    args = parse_args()

    with open(args.metadata) as f:
        metadata = json.load(f)

    use_lora = args.lora_weights is not None
    suffix = "lora" if use_lora else "base"
    root = Path(args.output_dir)

    log_dir = root / "logs"
    setup_logging(log_dir, f"{suffix}_c{args.caption_index}", args.device)

    # Log all configuration
    log.info("Configuration:")
    for k, v in sorted(vars(args).items()):
        log.info("  %-22s = %s", k, v)

    # ------------------------------------------------------------------
    # Subset metadata: sampling OR start/limit (mutually exclusive)
    # ------------------------------------------------------------------
    if args.train_pct is not None or args.val_pct is not None:
        ci = args.caption_index
        def _has_classes(e):
            cc = e.get("caption_classes")
            if cc and ci < len(cc):
                return bool(cc[ci].get("classes"))
            return bool(e.get("classes"))

        train_all = [e for e in metadata if e.get("split") == "train"]
        val_all = [e for e in metadata if e.get("split") == "val"]
        train_entries = [e for e in train_all if _has_classes(e)]
        val_entries = [e for e in val_all if _has_classes(e)]
        excluded = (len(train_all) - len(train_entries)) + (len(val_all) - len(val_entries))
        if excluded:
            log.info("Excluded %d images with empty classes (train: %d, val: %d)",
                     excluded, len(train_all) - len(train_entries), len(val_all) - len(val_entries))
        rng = random.Random(args.sample_seed)

        if args.train_pct is not None:
            train_entries = _stratified_sample(train_entries, args.train_pct, rng, args.caption_index)
        else:
            train_entries = []

        if args.val_pct is not None:
            val_entries = _stratified_sample(val_entries, args.val_pct, rng, args.caption_index)
        else:
            val_entries = []

        metadata = train_entries + val_entries
        log.info("Sampled %d train + %d val = %d total (seed=%d)",
                 len(train_entries), len(val_entries), len(metadata), args.sample_seed)
    else:
        metadata = metadata[args.start:]
        if args.limit:
            metadata = metadata[:args.limit]

    # ------------------------------------------------------------------
    # Build entries
    # ------------------------------------------------------------------

    entries = []
    caption_classes_used = 0
    for entry in metadata:
        img_id = Path(entry["image"]).stem
        caption = entry["captions"][args.caption_index]
        split = entry.get("split", "val")
        rns_folder = "train2017" if split == "train" else "val2017"

        # Use caption-specific classes; skip image if not available
        cc = entry.get("caption_classes")
        if not cc or args.caption_index >= len(cc):
            log.warning("No caption_classes for image %s — skipping", img_id)
            continue
        classes = cc[args.caption_index]["classes"]
        caption_classes_used += 1

        if "background" not in [c.lower() for c in classes]:
            classes = classes + ["background"]
        entries.append({"id": img_id, "caption": caption, "classes": classes,
                        "rns_folder": rns_folder})

    log.info("Caption-specific classes: %d/%d entries (fallback to GT for rest)",
             caption_classes_used, len(entries))

    images_base = root / f"generated_images_{suffix}_c{args.caption_index}"
    masks_base = root / f"generated_masks_{suffix}_c{args.caption_index}"
    for rns_folder in ("train2017", "val2017"):
        (images_base / rns_folder).mkdir(parents=True, exist_ok=True)
        (masks_base / rns_folder).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Multi-GPU dispatch
    # ------------------------------------------------------------------
    if args.num_gpus > 1:
        if args.train_pct is not None or args.val_pct is not None:
            log.error("--train_pct/--val_pct not supported with --num_gpus > 1. "
                      "Use single GPU or pre-sample metadata.")
            sys.exit(1)
        log.info("Multi-GPU mode: splitting %d images across %d GPUs", len(entries), args.num_gpus)
        log.info("Mode: %s", suffix)
        log.info("=" * 60)
        failed = _launch_multi_gpu(args, len(entries))
        log.info("=" * 60)
        if failed:
            log.warning("GPU workers %s failed!", failed)
        else:
            log.info("All workers finished successfully.")

        n_img = len(list(images_base.rglob("*.jpg")))
        n_mask = len(list(masks_base.rglob("*_instanceTrainIds.png")))
        log.info("  Images: %d  |  Masks: %d", n_img, n_mask)
        return

    # ------------------------------------------------------------------
    # Single-GPU path
    # ------------------------------------------------------------------
    lora_targets = [m.strip() for m in args.lora_targets.split(",")] if args.lora_targets else None
    attn_size = args.resolution // 16

    log.info("Mode:       %s", suffix)
    log.info("Processing: %d images", len(entries))
    log.info("Images ->   %s", images_base)
    log.info("Masks  ->   %s", masks_base)
    log.info("Resolution: %d", args.resolution)
    log.info("Device:     %s", args.device)
    log.info("LoRA:       %s", args.lora_weights or "None (base only)")
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # Load pipeline (+ LoRA if needed)
    # ------------------------------------------------------------------
    pipe = load_pipeline(args.model_id, fp16=args.fp16, device=args.device)
    if use_lora:
        pipe = apply_lora(
            pipe, args.lora_weights,
            lora_rank=args.lora_rank,
            target_modules=lora_targets,
            lora_layers=args.lora_layers,
            weight_prefix=args.lora_weight_prefix,
        )
    setup_attention_caching(pipe, args.attention_layers)

    # Background thread pool for I/O (saves while GPU works on next image)
    io_pool = ThreadPoolExecutor(max_workers=2)
    io_futures = []

    # ------------------------------------------------------------------
    # Generate + segment each image
    # ------------------------------------------------------------------
    t0 = time.time()
    processed = 0
    skipped = 0
    skipped_no_classes = 0
    ovss_count = 0
    unsup_count = 0
    gen_times = []
    seg_times = []
    total_classes_used = []

    for i, entry in enumerate(entries):
        img_id, caption, classes = entry["id"], entry["caption"], entry["classes"]
        rns_folder = entry["rns_folder"]
        img_path = images_base / rns_folder / f"{img_id}.jpg"
        mask_path = masks_base / rns_folder / f"{img_id}_instanceTrainIds.png"

        if args.skip_existing and img_path.exists() and mask_path.exists():
            log.info("[%d/%d] Skip %s (exists)", i + 1, len(entries), img_id)
            skipped += 1
            continue

        if all(c.lower() == "background" for c in classes):
            log.warning("[SKIP] No classes for image %s (background only)", img_id)
            skipped_no_classes += 1
            continue

        log.info("[%s] (%d/%d) %s: %s", suffix, i + 1, len(entries), img_id, caption[:70])

        # --- Generation ---
        t_gen = time.time()
        disable_attention_caching(pipe)
        gen_img = generate_image(
            pipe, caption,
            negative_prompt=args.negative_prompt,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            resolution=args.resolution,
        )
        gen_elapsed = time.time() - t_gen
        gen_times.append(gen_elapsed)

        # --- Segmentation ---
        t_seg = time.time()
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
                ovss_count += 1
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
                classes = [f"segment_{j}" for j in range(masks.shape[0])]
                unsup_count += 1
        seg_elapsed = time.time() - t_seg
        seg_times.append(seg_elapsed)
        total_classes_used.append(len(classes))

        io_futures.append(io_pool.submit(
            lambda p, im: im.save(p, format="JPEG", quality=95),
            str(img_path), gen_img))
        io_futures.append(io_pool.submit(save_mask, gen_img, masks, classes, mask_path))

        processed += 1
        elapsed = time.time() - t0
        avg = elapsed / processed
        remaining = avg * (len(entries) - i - 1)
        log.info("  gen=%.1fs  seg=%.1fs  |  %ds elapsed, ~%ds remaining",
                 gen_elapsed, seg_elapsed, int(elapsed), int(remaining))

    # Wait for all background saves to finish
    for fut in io_futures:
        fut.result()
    io_pool.shutdown()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_elapsed = time.time() - t0
    n_img = len(list(images_base.rglob("*.jpg")))
    n_mask = len(list(masks_base.rglob("*_instanceTrainIds.png")))

    log.info("=" * 60)
    log.info("BATCH PIPELINE COMPLETE")
    log.info("=" * 60)
    log.info("  Mode:            %s", suffix)
    log.info("  Device:          %s", args.device)
    log.info("  Resolution:      %d", args.resolution)
    log.info("-" * 60)
    log.info("  Total entries:   %d", len(entries))
    log.info("  Processed:       %d", processed)
    log.info("  Skipped (exist): %d", skipped)
    log.info("  Skipped (no cls):%d", skipped_no_classes)
    log.info("-" * 60)
    log.info("  OVSS segments:   %d", ovss_count)
    log.info("  Unsup segments:  %d", unsup_count)
    if total_classes_used:
        log.info("  Avg classes/img: %.1f", sum(total_classes_used) / len(total_classes_used))
    log.info("-" * 60)
    if gen_times:
        log.info("  Generation time: %.1fs total  |  avg %.2fs  |  min %.2fs  |  max %.2fs",
                 sum(gen_times), sum(gen_times) / len(gen_times),
                 min(gen_times), max(gen_times))
    if seg_times:
        log.info("  Segment time:    %.1fs total  |  avg %.2fs  |  min %.2fs  |  max %.2fs",
                 sum(seg_times), sum(seg_times) / len(seg_times),
                 min(seg_times), max(seg_times))
    log.info("  Total wall time: %ds (%.1fs/image)",
             int(total_elapsed), total_elapsed / max(processed, 1))
    log.info("-" * 60)
    log.info("  Images on disk:  %s  (%d files)", images_base, n_img)
    log.info("  Masks on disk:   %s  (%d files)", masks_base, n_mask)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
