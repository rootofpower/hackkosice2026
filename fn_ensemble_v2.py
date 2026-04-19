#!/usr/bin/env python3
"""Ensemble evaluation script for v2 models — combines models trained with
   different seeds and/or backbones using TTA + threshold sweep."""

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

from sklearn.metrics import recall_score, precision_score, f1_score
from tqdm import tqdm

from config import PAIRS

IMAGE_SIZE   = (490, 490)
BATCH_SIZE   = 16
N_TTA        = 30
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

def load_data(pair_config: dict):
    def collect(disease_dir, healthy_dir, split_name):
        paths, labels = [], []
        for p in sorted(Path(disease_dir).glob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                paths.append(str(p))
                labels.append(0)
        for p in sorted(Path(healthy_dir).glob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                paths.append(str(p))
                labels.append(1)
        log.info(f"{split_name} — disease: {labels.count(0)}, "
                 f"healthy: {labels.count(1)}, "
                 f"total: {len(labels)}")
        return np.array(paths), np.array(labels)

    train_paths, train_labels = collect(
        pair_config["disease_train"], pair_config["healthy_train"], "Train"
    )
    test_paths, test_labels = collect(
        pair_config["disease_test"], pair_config["healthy_test"], "Test"
    )
    return train_paths, train_labels, test_paths, test_labels

def validate_paths(pair_key: str):
    pair = PAIRS[pair_key]
    for key, path in pair.items():
        if key == "label":
            continue
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"[{pair_key}] Path not found: {path}\n"
                f"  Key: {key}"
            )
        files = [f for f in p.glob("*")
                 if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]
        if len(files) == 0:
            raise ValueError(
                f"[{pair_key}] No images found in: {path}\n"
                f"  Key: {key}"
            )
        log.info(f"  {key}: {len(files)} images → {path}")

def build_model(backbone: str):
    """Build model matching the backbone used during training."""
    if backbone == "b0":
        model = models.efficientnet_b0(weights=None)
        in_features = 1280
    elif backbone == "b2":
        model = models.efficientnet_b2(weights=None)
        in_features = 1408
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, NUM_CLASSES)
    )
    return model

tta_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5)),
    transforms.ColorJitter(brightness=0.3, contrast=0.4),
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
    n = len(variants_data)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    if n == 1:
        axes = [axes]
    
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
    parser.add_argument("--backbone", default="b2", choices=["b0", "b2"],
                        help="Backbone used for v2 models")
    args = parser.parse_args()
    
    PAIR_KEY = args.pair
    PAIR = PAIRS[PAIR_KEY]
    BB = args.backbone
    
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    RESULTS_DIR = f"results/{PAIR_KEY}"
    
    # Model A: seed 42, Model B: seed 123
    MODEL_A = f"{RESULTS_DIR}/efficientnet_v2_{BB}_{PAIR_KEY}.pth"
    MODEL_B = f"{RESULTS_DIR}/efficientnet_v2_{BB}_s123_{PAIR_KEY}.pth"
    LOG_FILE = f"{RESULTS_DIR}/fn_ensemble_v2_{PAIR_KEY}.log"
    PLOT_FILE = f"{RESULTS_DIR}/ensemble_v2_threshold_sweep_{PAIR_KEY}.png"
    
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
    
    log.info(f"Starting fn_ensemble v2 pipeline for pair: {PAIR_KEY}")
    log.info(f"Label: {PAIR['label']}")
    log.info(f"Backbone: {BB}")
    
    validate_paths(PAIR_KEY)
    
    _, _, test_paths, test_labels = load_data(PAIR)
    
    # Check which models exist
    has_model_a = os.path.exists(MODEL_A)
    has_model_b = os.path.exists(MODEL_B)
    
    if not has_model_a:
        log.error(f"Model A not found: {MODEL_A}")
        return
    
    # Load model A
    model_a = build_model(BB)
    model_a = model_a.to(DEVICE)
    log.info(f"Loading Model A (seed=42): {MODEL_A}")
    model_a.load_state_dict(torch.load(MODEL_A, map_location=DEVICE))
    model_a.eval()
    
    # Load model B if available
    model_b = None
    if has_model_b:
        model_b = build_model(BB)
        model_b = model_b.to(DEVICE)
        log.info(f"Loading Model B (seed=123): {MODEL_B}")
        model_b.load_state_dict(torch.load(MODEL_B, map_location=DEVICE))
        model_b.eval()
    else:
        log.info(f"Model B not found ({MODEL_B}), running single-model TTA only.")
    
    # ---- Also try loading old models for backward-compatible ensemble ----
    OLD_MODEL_FT = f"{RESULTS_DIR}/efficientnet_finetuned_{PAIR_KEY}.pth"
    OLD_MODEL_FOCAL = f"{RESULTS_DIR}/efficientnet_focal_{PAIR_KEY}.pth"
    
    old_model = None
    if os.path.exists(OLD_MODEL_FT):
        old_model = models.efficientnet_b0(weights=None)
        old_model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1280, NUM_CLASSES)
        )
        old_model = old_model.to(DEVICE)
        log.info(f"Loading Old Model (finetune B0): {OLD_MODEL_FT}")
        old_model.load_state_dict(torch.load(OLD_MODEL_FT, map_location=DEVICE))
        old_model.eval()
    
    y_true = []
    y_probs_single = []   # V1: single model A TTA
    y_probs_v2 = []       # V2: average of A + B (if B exists)
    y_probs_v3 = []       # V3: average of v2_A + old_model (if exists)
    
    models_list = [m for m in [model_a, model_b, old_model] if m is not None]
    model_names = []
    if model_a: model_names.append("v2_A")
    if model_b: model_names.append("v2_B")
    if old_model: model_names.append("old_FT")
    
    log.info(f"Ensemble models: {model_names}")
    
    for i in tqdm(range(len(test_paths)), desc="Ensemble Inference"):
        path = test_paths[i]
        label = test_labels[i]
        try:
            image_pil = Image.open(path).convert('L')
        except:
            image_pil = Image.fromarray(np.zeros(IMAGE_SIZE, dtype=np.uint8))
        
        prob_a = predict_tta(model_a, image_pil, n=N_TTA)
        
        y_true.append(label)
        y_probs_single.append(prob_a[0, 0].item())
        
        if model_b is not None:
            prob_b = predict_tta(model_b, image_pil, n=N_TTA)
            prob_avg_ab = (prob_a + prob_b) / 2
            y_probs_v2.append(prob_avg_ab[0, 0].item())
        
        if old_model is not None:
            prob_old = predict_tta(old_model, image_pil, n=N_TTA)
            prob_avg_cross = (prob_a + prob_old) / 2
            y_probs_v3.append(prob_avg_cross[0, 0].item())
        
    y_true = np.array(y_true)
    y_probs_single = np.array(y_probs_single)
    
    # Evaluate all available variants
    all_points = []
    all_plot_data = []
    
    pts_v1, th_v1, r_v1, p_v1, f1_v1 = evaluate_thresholds(
        y_true, y_probs_single, "Single Model A + TTA", 1)
    all_points.append(("V1 (Single+TTA)", pts_v1))
    all_plot_data.append(("V1: Single+TTA", th_v1, r_v1, p_v1, f1_v1, pts_v1))
    
    if len(y_probs_v2) > 0:
        y_probs_v2 = np.array(y_probs_v2)
        pts_v2, th_v2, r_v2, p_v2, f1_v2 = evaluate_thresholds(
            y_true, y_probs_v2, "Avg(v2_A + v2_B)", 2)
        all_points.append(("V2 (A+B Avg)", pts_v2))
        all_plot_data.append(("V2: A+B Avg", th_v2, r_v2, p_v2, f1_v2, pts_v2))
    
    if len(y_probs_v3) > 0:
        y_probs_v3 = np.array(y_probs_v3)
        pts_v3, th_v3, r_v3, p_v3, f1_v3 = evaluate_thresholds(
            y_true, y_probs_v3, "Avg(v2_A + old_FT)", 3)
        all_points.append(("V3 (v2+old)", pts_v3))
        all_plot_data.append(("V3: v2+old", th_v3, r_v3, p_v3, f1_v3, pts_v3))
    
    plot_ensemble(all_plot_data, PLOT_FILE)
    
    # Summary table
    log.info("=" * 80)
    log.info("ENSEMBLE COMPARISON — target: FN=0, minimize FP")
    log.info("=" * 80)
    log.info(f"{'Strategy':<35} {'t':>6} {'Recall':>8} {'Prec':>8} {'FN':>5} {'FP':>5} {'F1':>8}")
    
    def log_row(strat_name, pt):
        if pt:
            t, r, p, fn, fp, f1 = pt
            log.info(f"{strat_name:<35} {t:>6.2f} {r:>8.3f} {p:>8.3f} {fn:>5} {fp:>5} {f1:>8.3f}")
            
    for name, pts in all_points:
        log_row(f"{name} Point A", pts['A'])
        log_row(f"{name} Point B", pts['B'])
        log_row(f"{name} Point C", pts['C'])

if __name__ == "__main__":
    main()
