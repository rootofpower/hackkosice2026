#!/usr/bin/env python3
"""Ensemble evaluation script combining Finetuned and Focal models with TTA."""

import os
import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score
from tqdm import tqdm

from config import PAIRS

DATA_ROOT    = "./test"
IMAGE_SIZE   = (490, 490)
BATCH_SIZE   = 16
N_TTA        = 30
TRAIN_SPLIT  = 0.80
RANDOM_SEED  = 42
NUM_CLASSES  = 2
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log = logging.getLogger("main")

SUPPORTED_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

class TearDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(path).convert('L')
        except Exception as e:
            log.info(f"Error loading image {path}: {e}")
            image = Image.fromarray(np.zeros(IMAGE_SIZE, dtype=np.uint8))
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def _find_image_dir(class_dir: Path) -> Path:
    direct_images = [
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if direct_images:
        return class_dir

    for sub in sorted(class_dir.iterdir()):
        if sub.is_dir():
            sub_images = [
                p for p in sub.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
            if sub_images:
                return sub

    raise FileNotFoundError(
        f"No image files found in {class_dir} or its immediate subdirectories."
    )

def load_data(data_root: str, pair_config: dict):
    root_path = Path(data_root)
    classes = [
        {"label": 0, "name": pair_config["disease_class"]},
        {"label": 1, "name": pair_config["healthy_class"]}
    ]
    
    paths = []
    labels = []
    
    for cls in classes:
        class_dir = root_path / cls["name"]
        if not class_dir.exists():
            log.info(f"Warning: class directory not found: {class_dir}")
            continue
            
        try:
            image_dir = _find_image_dir(class_dir)
        except FileNotFoundError as e:
            log.info(f"Warning: {e}")
            continue
            
        cls_paths = sorted(
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        for p in cls_paths:
            paths.append(str(p))
            labels.append(cls["label"])
            
    return np.array(paths), np.array(labels)

def get_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(1280, NUM_CLASSES)
    )
    return model

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

def predict_tta(model, image_pil, n=N_TTA):
    preds = []
    for _ in range(n):
        aug = tta_transform(image_pil)
        with torch.no_grad():
            logits = model(aug.unsqueeze(0).to(DEVICE))
            preds.append(torch.softmax(logits, dim=1).cpu())
    return torch.stack(preds).mean(0)  # shape: (1, 2)

def evaluate_thresholds(y_true, probs_disease, name, v):
    thresholds = np.arange(0.20, 0.71, 0.01)
    recalls, precisions, f1s, fns, fps = [], [], [], [], []
    
    for t in thresholds:
        preds = np.where(probs_disease >= t, 0, 1)
        r = recall_score(y_true, preds, labels=[0, 1], pos_label=0, zero_division=0)
        p = precision_score(y_true, preds, labels=[0, 1], pos_label=0, zero_division=0)
        f1 = f1_score(y_true, preds, average='macro', zero_division=0)
        fn = np.sum((y_true == 0) & (preds == 1))
        fp = np.sum((y_true == 1) & (preds == 0))
        
        recalls.append(r)
        precisions.append(p)
        f1s.append(f1)
        fns.append(fn)
        fps.append(fp)
        
    recalls = np.array(recalls)
    precisions = np.array(precisions)
    f1s = np.array(f1s)
    thresholds = np.array(thresholds)
    
    idx_a = np.argmax(recalls)
    
    valid = [(t, r, p, fn, fp, f1) for t, r, p, fn, fp, f1 in zip(thresholds, recalls, precisions, fns, fps, f1s) if r >= 0.95]
    if valid:
        point_b = max(valid, key=lambda x: x[2])  # maximize precision
    else:
        point_b = None
        
    idx_c = np.argmax(f1s)
    
    log.info(f"=== Ensemble Variant {v}: {name} ===")
    log.info("Point A (Maximize Recall):")
    log.info(f"  Threshold:         {thresholds[idx_a]:.2f}")
    log.info(f"  Disease recall:    {recalls[idx_a]:.4f}")
    log.info(f"  Disease precision: {precisions[idx_a]:.4f}")
    log.info(f"  False negatives:   {fns[idx_a]}")
    log.info(f"  False positives:   {fps[idx_a]}")
    log.info(f"  F1-macro:          {f1s[idx_a]:.4f}")
    
    if point_b:
        log.info("Point B (Recall >= 0.95 best precision):")
        log.info(f"  Threshold:         {point_b[0]:.2f}")
        log.info(f"  Disease recall:    {point_b[1]:.4f}")
        log.info(f"  Disease precision: {point_b[2]:.4f}")
        log.info(f"  False negatives:   {point_b[3]}")
        log.info(f"  False positives:   {point_b[4]}")
        log.info(f"  F1-macro:          {point_b[5]:.4f}")
        pb_data = point_b
    else:
        log.info("Point B (Recall >= 0.95 best precision): None found.")
        pb_data = None
        
    log.info("Point C (Maximize F1-macro):")
    log.info(f"  Threshold:         {thresholds[idx_c]:.2f}")
    log.info(f"  Disease recall:    {recalls[idx_c]:.4f}")
    log.info(f"  Disease precision: {precisions[idx_c]:.4f}")
    log.info(f"  False negatives:   {fns[idx_c]}")
    log.info(f"  False positives:   {fps[idx_c]}")
    log.info(f"  F1-macro:          {f1s[idx_c]:.4f}")
    
    points = {
        'A': (thresholds[idx_a], recalls[idx_a], precisions[idx_a], fns[idx_a], fps[idx_a], f1s[idx_a]),
        'B': pb_data,
        'C': (thresholds[idx_c], recalls[idx_c], precisions[idx_c], fns[idx_c], fps[idx_c], f1s[idx_c])
    }
    
    return points, thresholds, recalls, precisions, f1s

def plot_ensemble(variants_data, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (name, thresh, rec, prec, f1s, pts) in enumerate(variants_data):
        ax = axes[idx]
        ax.plot(thresh, rec, label='Recall', color='blue')
        ax.plot(thresh, prec, label='Precision', color='orange')
        ax.plot(thresh, f1s, label='F1-macro', color='green')
        
        if pts['A']:
            ax.axvline(pts['A'][0], color='blue', linestyle='--', label=f"Pt A (t={pts['A'][0]:.2f})")
        if pts['B']:
            ax.axvline(pts['B'][0], color='orange', linestyle='--', label=f"Pt B (t={pts['B'][0]:.2f})")
        if pts['C']:
            ax.axvline(pts['C'][0], color='green', linestyle='--', label=f"Pt C (t={pts['C'][0]:.2f})")
            
        ax.set_title(name)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.legend()
        ax.grid(True)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True, choices=list(PAIRS.keys()))
    args = parser.parse_args()
    
    PAIR_KEY = args.pair
    PAIR = PAIRS[PAIR_KEY]
    
    RESULTS_DIR = f"results/{PAIR_KEY}"
    MODEL_A = f"{RESULTS_DIR}/efficientnet_finetuned_{PAIR_KEY}.pth"
    MODEL_B = f"{RESULTS_DIR}/efficientnet_focal_{PAIR_KEY}.pth"
    LOG_FILE = f"{RESULTS_DIR}/fn_ensemble_{PAIR_KEY}.log"
    PLOT_FILE = f"{RESULTS_DIR}/ensemble_threshold_sweep_{PAIR_KEY}.png"
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Configure logging dynamically
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    
    log.info(f"Starting fn_ensemble pipeline for pair: {PAIR_KEY}")
    
    data_path = f"{DATA_ROOT}/{PAIR_KEY}"
    paths, labels = load_data(data_path, PAIR)
    indices = np.arange(len(paths))
    train_idx, test_idx = train_test_split(
        indices,
        train_size=TRAIN_SPLIT,
        stratify=labels,
        random_state=RANDOM_SEED
    )
    
    test_paths = paths[test_idx]
    test_labels = labels[test_idx]
    
    model_a = get_model()
    model_a = model_a.to(DEVICE)
    log.info(f"Loading Model A: {MODEL_A}")
    model_a.load_state_dict(torch.load(MODEL_A, map_location=DEVICE))
    model_a.eval()
    
    model_b = get_model()
    model_b = model_b.to(DEVICE)
    log.info(f"Loading Model B: {MODEL_B}")
    model_b.load_state_dict(torch.load(MODEL_B, map_location=DEVICE))
    model_b.eval()
    
    y_true = []
    y_probs_v1 = []
    y_probs_v2 = []
    y_probs_v3 = []
    
    for i in tqdm(range(len(test_paths)), desc="Ensemble Inference"):
        path = test_paths[i]
        label = test_labels[i]
        try:
            image_pil = Image.open(path).convert('L')
        except:
            image_pil = Image.fromarray(np.zeros(IMAGE_SIZE, dtype=np.uint8))
        
        prob_a = predict_tta(model_a, image_pil, n=N_TTA)
        prob_b = predict_tta(model_b, image_pil, n=N_TTA)
        
        prob_avg = (prob_a + prob_b) / 2
        prob_weighted = 0.4 * prob_a + 0.6 * prob_b
        prob_max = torch.stack([prob_a, prob_b]).max(dim=0).values
        
        y_true.append(label)
        y_probs_v1.append(prob_avg[0, 0].item())
        y_probs_v2.append(prob_weighted[0, 0].item())
        y_probs_v3.append(prob_max[0, 0].item())
        
    y_true = np.array(y_true)
    y_probs_v1 = np.array(y_probs_v1)
    y_probs_v2 = np.array(y_probs_v2)
    y_probs_v3 = np.array(y_probs_v3)
    
    pts_v1, th_v1, r_v1, p_v1, f1_v1 = evaluate_thresholds(y_true, y_probs_v1, "Simple Average", 1)
    pts_v2, th_v2, r_v2, p_v2, f1_v2 = evaluate_thresholds(y_true, y_probs_v2, "Weighted Average (0.4A+0.6B)", 2)
    pts_v3, th_v3, r_v3, p_v3, f1_v3 = evaluate_thresholds(y_true, y_probs_v3, "Max Probability", 3)
    
    plot_data = [
        ("V1: Simple Average", th_v1, r_v1, p_v1, f1_v1, pts_v1),
        ("V2: Weighted Average", th_v2, r_v2, p_v2, f1_v2, pts_v2),
        ("V3: Max Probability", th_v3, r_v3, p_v3, f1_v3, pts_v3)
    ]
    plot_ensemble(plot_data, PLOT_FILE)
    
    log.info("=" * 80)
    log.info("ENSEMBLE COMPARISON — target: FN=0, minimize FP")
    log.info("=" * 80)
    log.info(f"{'Strategy':<35} {'t':>6} {'Recall':>8} {'Prec':>8} {'FN':>5} {'FP':>5} {'F1':>8}")
    log.info(f"{'Baseline (single, t=0.50)':<35} {'0.50':>6} {'0.874':>8} {'0.974':>8} {'11':>5} {'2':>5} {'0.925':>8}")
    log.info(f"{'TTAx10 best (prev run)':<35} {'0.20':>6} {'0.977':>8} {'0.867':>8} {'2':>5} {'13':>5} {'0.913':>8}")
    
    def log_row(strat_name, pt):
        if pt:
            t, r, p, fn, fp, f1 = pt
            log.info(f"{strat_name:<35} {t:>6.2f} {r:>8.3f} {p:>8.3f} {fn:>5} {fp:>5} {f1:>8.3f}")
            
    log_row("V1 (Avg) Point A", pts_v1['A'])
    log_row("V1 (Avg) Point B", pts_v1['B'])
    log_row("V1 (Avg) Point C", pts_v1['C'])
    
    log_row("V2 (Weighted) Point A", pts_v2['A'])
    log_row("V2 (Weighted) Point B", pts_v2['B'])
    log_row("V2 (Weighted) Point C", pts_v2['C'])
    
    log_row("V3 (Max) Point A", pts_v3['A'])
    log_row("V3 (Max) Point B", pts_v3['B'])
    log_row("V3 (Max) Point C", pts_v3['C'])

if __name__ == "__main__":
    main()
