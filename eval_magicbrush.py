#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate InstructPix2Pix on osunlp/MagicBrush using:
- CLIP-T: sim(CLIP_vis(pred), CLIP_txt(gt_caption))
- CLIP-I: sim(CLIP_vis(pred), CLIP_vis(gt))
- DINO-I: sim(DINOv2(pred), DINOv2(gt))
"""
import argparse
import json
import random
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
from tqdm.auto import tqdm
from transformers import (
    CLIPModel,
    CLIPProcessor,
    AutoImageProcessor,
    AutoModel,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate InstructPix2Pix on MagicBrush")
    parser.add_argument("--model_id", type=str, default="timbrooks/instruct-pix2pix", help="HF model id for the base InstructPix2Pix pipeline")
    parser.add_argument("--lora_weights", type=str, default=None, help="Optional LoRA weights path (or hub repo) to load")
    parser.add_argument("--lora_weight_name", type=str, default=None, help="Optional weight_name when loading safetensors/ckpt")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate (train/validation/test)")
    parser.add_argument("--data_root", type=str, default="instructclip_datasets/MagicBrush", help="Path to local MagicBrush root directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32", "bf16"], help="Computation dtype for diffusion/encoders")
    parser.add_argument("--batch_size", type=int, default=1, help="Generation batch size (pipeline is not fully batched; keep small)")
    parser.add_argument("--save_dir", type=str, default=None, help="Optional directory to save predicted images")
    parser.add_argument("--csv_path", type=str, default="magicbrush_eval.csv", help="Where to save per-sample metrics")
    args = parser.parse_args()
    return args


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dtype(name: str):
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def _resolve_image_path(images_root: Path, filename: str) -> Path:
    """Locate an image file that might be nested under an ID subfolder."""
    direct_path = images_root / filename
    if direct_path.exists():
        return direct_path

    prefix = filename.split("-")[0]
    nested_path = images_root / prefix / filename
    if nested_path.exists():
        return nested_path

    raise FileNotFoundError(f"Could not find {filename} under {images_root}")


def load_local_magicbrush(data_root: str, split: str, seed: int):
    """Load MagicBrush samples from a local directory instead of HuggingFace hub."""
    json_path = Path(data_root) / split / "edit_turns.json"
    images_root = Path(data_root) / split / "images"
    captions_path = Path(data_root) / split / "global_descriptions.json"

    with open(json_path, "r") as f:
        records = json.load(f)

    captions = {}
    if captions_path.exists():
        with open(captions_path, "r") as f:
            raw = json.load(f)
        for _, group in raw.items():
            if isinstance(group, dict):
                for fname, desc in group.items():
                    captions[fname] = desc

    rng = random.Random(seed)
    rng.shuffle(records)

    dataset = []
    for rec in records:
        target_fname = rec["output"]
        caption = captions.get(target_fname)
        if caption is None:
            raise KeyError(f"Missing caption for {target_fname} in {captions_path}")
        dataset.append(
            {
                "instruction": rec["instruction"],
                "source_img_path": _resolve_image_path(images_root, rec["input"]),
                "target_img_path": _resolve_image_path(images_root, rec["output"]),
                "caption": caption,
            }
        )

    return dataset


def prepare_models(args):
    weight_dtype = get_dtype(args.dtype)
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(args.model_id, torch_dtype=weight_dtype)
    # pipe.load_lora_weights("SherryXTChen/InstructCLIP-InstructPix2Pix")
    if args.lora_weights:
        pipe.load_lora_weights(args.lora_weights, weight_name=args.lora_weight_name)
    pipe.to(args.device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None  # prevent NSFW alert

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(args.device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large", use_fast=True)
    dino_model = AutoModel.from_pretrained("facebook/dinov2-large").to(args.device)

    return pipe, clip_model, clip_processor, dino_model, dino_processor


def compute_clip_image_feats(clip_model, clip_processor, images: List[Image.Image], device: str):
    inputs = clip_processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = clip_model.get_image_features(**inputs)
    feats = F.normalize(feats, dim=-1)
    return feats


def compute_clip_text_feats(clip_model, clip_processor, captions: List[str], device: str):
    inputs = clip_processor(text=captions, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        feats = clip_model.get_text_features(**inputs)
    feats = F.normalize(feats, dim=-1)
    return feats


def compute_dino_feats(dino_model, dino_processor, images: List[Image.Image], device: str):
    inputs = dino_processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino_model(**inputs)
        # Use CLS token as global feature
        feats = outputs.last_hidden_state[:, 0]
    feats = F.normalize(feats, dim=-1)
    return feats


def save_image(img: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main():
    args = parse_args()
    set_seed(args.seed)

    pipe, clip_model, clip_processor, dino_model, dino_processor = prepare_models(args)
    ds = load_local_magicbrush(args.data_root, args.split, args.seed)

    pred_images = []
    gt_images = []
    prompts = []  # edit instructions
    captions = []  # reference descriptions of target images

    save_dir = Path(args.save_dir) if args.save_dir else None

    for idx in tqdm(range(len(ds)), desc="Generating", ncols=100):
        sample = ds[idx]
        prompt = sample["instruction"]
        orig_img = Image.open(sample["source_img_path"]).convert("RGB")
        gt_img = Image.open(sample["target_img_path"]).convert("RGB")

        with torch.autocast(device_type=args.device, enabled=args.dtype == "fp16"):
            pred = pipe(prompt, image=orig_img, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7).images[0]
        if save_dir:
            save_image(pred, save_dir / Path(sample["target_img_path"]).name)

        pred_images.append(pred)
        gt_images.append(gt_img)
        prompts.append(prompt)
        captions.append(sample["caption"])

    # Compute metrics in small batches to save memory
    clip_t_scores: List[float] = []
    clip_i_scores: List[float] = []
    dino_i_scores: List[float] = []

    bs = max(1, args.batch_size)
    for start in tqdm(range(0, len(pred_images), bs), desc="Scoring", ncols=100):
        end = start + bs
        batch_pred = pred_images[start:end]
        batch_gt = gt_images[start:end]
        batch_captions = captions[start:end]

        clip_img_pred = compute_clip_image_feats(clip_model, clip_processor, batch_pred, args.device)
        clip_img_gt = compute_clip_image_feats(clip_model, clip_processor, batch_gt, args.device)
        clip_txt = compute_clip_text_feats(clip_model, clip_processor, batch_captions, args.device)

        dino_pred = compute_dino_feats(dino_model, dino_processor, batch_pred, args.device)
        dino_gt = compute_dino_feats(dino_model, dino_processor, batch_gt, args.device)

        clip_t_scores.extend(F.cosine_similarity(clip_img_pred, clip_txt).cpu().tolist())
        clip_i_scores.extend(F.cosine_similarity(clip_img_pred, clip_img_gt).cpu().tolist())
        dino_i_scores.extend(F.cosine_similarity(dino_pred, dino_gt).cpu().tolist())

    import csv

    with open(args.csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "instruction", "caption", "clip_t", "clip_i", "dino_i"])
        for i, (p, c, t, ci, di) in enumerate(zip(prompts, captions, clip_t_scores, clip_i_scores, dino_i_scores)):
            writer.writerow([i, p, c, t, ci, di])

    print("==== Summary ====")
    print(f"CLIP-T: {sum(clip_t_scores)/len(clip_t_scores):.3f}")
    print(f"CLIP-I: {sum(clip_i_scores)/len(clip_i_scores):.3f}")
    print(f"DINO-I: {sum(dino_i_scores)/len(dino_i_scores):.3f}")
    print(f"Saved per-sample metrics to {args.csv_path}")
    if save_dir:
        print(f"Saved predictions to {save_dir}")


if __name__ == "__main__":
    main()
