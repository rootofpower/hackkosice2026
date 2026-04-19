#!/usr/bin/env python3
"""Minimal single-image classifier for the chory model pair."""

import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_A   = "results/chory/efficientnet_finetuned_chory.pth"
MODEL_B   = "results/chory/efficientnet_focal_chory.pth"
THRESHOLD = 0.48
N_TTA     = 10
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def build_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(1280, 2)
    )
    return model

def predict_tta(model, image_pil):
    preds = []
    for _ in range(N_TTA):
        aug = tta_transform(image_pil)
        with torch.no_grad():
            logits = model(aug.unsqueeze(0).to(DEVICE))
            preds.append(torch.softmax(logits, dim=1).cpu())
    return torch.stack(preds).mean(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    image_pil = Image.open(args.image).convert("L")

    model_a = build_model()
    model_a.load_state_dict(torch.load(MODEL_A, map_location=DEVICE))
    model_a.to(DEVICE).eval()

    model_b = build_model()
    model_b.load_state_dict(torch.load(MODEL_B, map_location=DEVICE))
    model_b.to(DEVICE).eval()

    prob_a = predict_tta(model_a, image_pil)
    prob_b = predict_tta(model_b, image_pil)
    prob = (prob_a + prob_b) / 2

    disease_prob = prob[0, 0].item()
    prediction = "SICK" if disease_prob >= THRESHOLD else "HEALTHY"

    distance = abs(disease_prob - THRESHOLD)
    if distance > 0.20:
        confidence = "high"
    elif distance > 0.10:
        confidence = "medium"
    else:
        confidence = "low"

    if prediction == "SICK":
        result_str = "\033[91mSICK\033[0m"
    else:
        result_str = "\033[92mHEALTHY\033[0m"

    import os
    file_name = os.path.basename(args.image)
    if len(file_name) > 20:
        file_name = file_name[:17] + "..."

    dp_str = f"{disease_prob*100:.1f}%"

    print("╔══════════════════════════════════════╗")
    print("║     LacriML — Chory vs Zdravi        ║")
    print("╠══════════════════════════════════════╣")
    print(f"║  File:         {file_name:<21}║")
    print(f"║  Result:       {result_str:<30}║")
    print(f"║  Disease prob: {dp_str:<21}║")
    print(f"║  Confidence:   {confidence:<21}║")
    print("╚══════════════════════════════════════╝")

if __name__ == "__main__":
    main()
