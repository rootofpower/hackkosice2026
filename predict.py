#!/usr/bin/env python3
# Usage example: python predict.py --image path/to/image.png --mode balanced
"""
LacriML Standalone Inference Script

Provides both a CLI and an importable module for classifying tear microscopy images.
Uses an ensemble of Finetuned and Focal EfficientNet-B0 models with Test-Time Augmentation (TTA).
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# --- Configuration ---
MODEL_A    = "results/efficientnet_finetuned.pth"
MODEL_B    = "results/efficientnet_focal.pth"
N_TTA      = 30
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (490, 490)

THRESHOLDS = {
    "screening":  0.20,   # FN=0,  FP=26 — catch every sick patient
    "balanced":   0.42,   # FN=1,  FP=1  — recommended default
    "precision":  0.49,   # FN=11, FP=2  — minimize false alarms
}
DEFAULT_MODE = "balanced"

# --- Architecture ---
def build_model():
    """Build the EfficientNet-B0 model architecture."""
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(1280, 2)
    )
    return model

# --- Data Transformations ---
tta_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.ColorJitter(brightness=0.2, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Model Loading Cache ---
_models = None

def _load_models():
    """Load both EfficientNet models and cache them in memory."""
    global _models
    if _models is not None:
        return _models
        
    if not os.path.exists(MODEL_A):
        raise FileNotFoundError(f"Missing Model A checkpoint at expected path: {MODEL_A}")
    if not os.path.exists(MODEL_B):
        raise FileNotFoundError(f"Missing Model B checkpoint at expected path: {MODEL_B}")
        
    model_a = build_model()
    model_a.load_state_dict(torch.load(MODEL_A, map_location=DEVICE))
    model_a.to(DEVICE).eval()
    
    model_b = build_model()
    model_b.load_state_dict(torch.load(MODEL_B, map_location=DEVICE))
    model_b.to(DEVICE).eval()
    
    _models = (model_a, model_b)
    return _models

def _predict_tta(model, image_pil, n=N_TTA):
    """Run TTA inference for a single model."""
    preds = []
    for _ in range(n):
        aug = tta_transform(image_pil)
        with torch.no_grad():
            logits = model(aug.unsqueeze(0).to(DEVICE))
            preds.append(torch.softmax(logits, dim=1).cpu())
    return torch.stack(preds).mean(0)  # shape: (1, 2)

# --- Core Inference Functions ---
def predict(image_path: str, mode: str = DEFAULT_MODE) -> dict:
    """
    Classify a single tear microscopy image.

    Args:
        image_path: path to grayscale microscopy image (490x490 px)
        mode: "screening" | "balanced" | "precision"

    Returns:
        {
            "file":         str,     # filename
            "prediction":   "SICK" or "HEALTHY",
            "disease_prob": float,   # probability of disease (0-1)
            "healthy_prob": float,
            "confidence":   "high" | "medium" | "low",
            "mode":         str,
            "threshold":    float,
        }
    """
    if mode not in THRESHOLDS:
        raise ValueError(f"Unknown mode: {mode}. Valid modes are: {list(THRESHOLDS.keys())}")
        
    threshold = THRESHOLDS[mode]
    
    try:
        image_pil = Image.open(image_path).convert('L')
    except Exception as e:
        print(f"Warning: Failed to load image {image_path}: {e}")
        return None
        
    if image_pil.size != IMAGE_SIZE:
        print(f"Warning: Image size {image_pil.size} differs from expected {IMAGE_SIZE}.")
        
    model_a, model_b = _load_models()
    
    prob_a = _predict_tta(model_a, image_pil)
    prob_b = _predict_tta(model_b, image_pil)
    
    # Variant 3: Max probability (most aggressive recall)
    prob_max = torch.stack([prob_a, prob_b]).max(dim=0).values
    
    disease_prob = float(prob_max[0, 0].item())
    healthy_prob = float(prob_max[0, 1].item())
    
    prediction = "SICK" if disease_prob >= threshold else "HEALTHY"
    
    distance = abs(disease_prob - threshold)
    if distance > 0.20:
        confidence = "high"
    elif distance > 0.10:
        confidence = "medium"
    else:
        confidence = "low"
        
    return {
        "file": os.path.basename(image_path),
        "prediction": prediction,
        "disease_prob": disease_prob,
        "healthy_prob": healthy_prob,
        "confidence": confidence,
        "mode": mode,
        "threshold": threshold
    }

def predict_batch(image_paths: list, mode: str = DEFAULT_MODE) -> list:
    """Run predict() on a list of image paths. Returns list of result dicts."""
    results = []
    for path in tqdm(image_paths, desc="Classifying"):
        res = predict(path, mode=mode)
        if res is not None:
            results.append(res)
    return results

# --- CLI Handlers ---
def print_single_result(res):
    """Print nicely formatted box for single image result."""
    file_name = res['file']
    if len(file_name) > 20: 
        file_name = file_name[:17] + "..."
        
    dp_str = f"{res['disease_prob']*100:.1f}%"
    mode_str = f"{res['mode']} (t={res['threshold']:.2f})"
    
    print("╔══════════════════════════════════════╗")
    print("║         LacriML — Prediction         ║")
    print("╠══════════════════════════════════════╣")
    print(f"║  File:        {file_name:<22} ║")
    print(f"║  Result:      {res['prediction']:<22} ║")
    print(f"║  Disease prob:{dp_str:<23} ║")
    print(f"║  Confidence:  {res['confidence']:<22} ║")
    print(f"║  Mode:        {mode_str:<22} ║")
    print("╚══════════════════════════════════════╝")

def print_batch_summary(results):
    """Print statistics summary for a batch execution."""
    if not results:
        print("No valid results to summarize.")
        return
        
    total = len(results)
    sick = sum(1 for r in results if r['prediction'] == "SICK")
    healthy = total - sick
    high_conf = sum(1 for r in results if r['confidence'] == "high")
    
    print(f"Total:    {total} images")
    print(f"Sick:     {sick}  ({sick/total*100:.1f}%)")
    print(f"Healthy:  {healthy}  ({healthy/total*100:.1f}%)")
    print(f"High confidence predictions: {high_conf/total*100:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LacriML tear image classifier")
    parser.add_argument("--image",  required=True,  help="Path to image or directory")
    parser.add_argument("--mode",   default=DEFAULT_MODE,
                        choices=list(THRESHOLDS.keys()),
                        help="Operating mode (default: balanced)")
    parser.add_argument("--output", default=None,
                        help="Optional: save results as JSON to this path")
    args = parser.parse_args()
    
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Path {image_path} does not exist.")
        sys.exit(1)
        
    if image_path.is_file():
        res = predict(str(image_path), mode=args.mode)
        if res:
            print_single_result(res)
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump([res], f, indent=2)
                print(f"\nResults saved to {args.output}")
    elif image_path.is_dir():
        # find all supported images
        extensions = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        images = []
        for p in image_path.rglob("*"):
            if p.is_file() and p.suffix.lower() in extensions:
                images.append(str(p))
                
        if not images:
            print(f"No valid images found in {image_path}.")
            sys.exit(0)
            
        print(f"Found {len(images)} images in directory. Starting batch classification...")
        results = predict_batch(images, mode=args.mode)
        print("\n")
        print_batch_summary(results)
        
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")
