#!/usr/bin/env python3
"""Evaluation script to reduce False Negatives using Thresholds, TTA and Focal Loss."""

import os
import logging
import copy
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score
)
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

DATA_ROOT    = "./test"
RESULTS_DIR  = "results"
LOG_FILE     = "results/fn_reduction_tta_no_scale.log"
IMAGE_SIZE   = (490, 490)
BATCH_SIZE   = 16
N_TTA        = 10
TRAIN_SPLIT  = 0.80
RANDOM_SEED  = 42
NUM_CLASSES  = 2
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FINETUNED_CKPT = "results/efficientnet_finetuned.pth"
FOCAL_CKPT     = "results/efficientnet_focal.pth"
FOCAL_LOG      = "results/efficientnet_focal.log"
FOCAL_EPOCHS   = 20
FOCAL_PATIENCE = 8

os.makedirs(RESULTS_DIR, exist_ok=True)

# Main Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("main")

# Focal Logger
focal_logger = logging.getLogger("focal")
focal_logger.setLevel(logging.INFO)
focal_logger.propagate = False
fh = logging.FileHandler(FOCAL_LOG, mode="w", encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
focal_logger.addHandler(fh)
focal_logger.addHandler(sh)


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

def load_data(data_root: str):
    root_path = Path(data_root)
    classes = [
        {"label": 0, "name": "diabetes_spolu_zo_suchym_okom"},
        {"label": 1, "name": "zdravi_ludi"}
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

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss
        return focal.mean() if self.reduction == 'mean' else focal

def evaluate_thresholds(y_true, probs_disease, name="Strategy 1"):
    thresholds = np.arange(0.20, 0.71, 0.01)
    recalls = []
    precisions = []
    f1s = []
    fns = []
    fps = []
    
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
    
    valid = [(r, p, t, fn, fp, f1) for r, p, t, fn, fp, f1 in zip(recalls, precisions, thresholds, fns, fps, f1s) if r >= 0.95]
    if valid:
        point_b = max(valid, key=lambda x: x[1])
    else:
        point_b = None
        
    idx_c = np.argmax(f1s)
    
    log.info(f"=== {name} ===")
    log.info("Point A (Maximize Recall):")
    log.info(f"  Threshold:         {thresholds[idx_a]:.2f}")
    log.info(f"  Disease recall:    {recalls[idx_a]:.4f}")
    log.info(f"  Disease precision: {precisions[idx_a]:.4f}")
    log.info(f"  False negatives:   {fns[idx_a]}")
    log.info(f"  False positives:   {fps[idx_a]}")
    log.info(f"  F1-macro:          {f1s[idx_a]:.4f}")
    
    if point_b:
        log.info("Point B (Recall >= 0.95 best precision):")
        log.info(f"  Threshold:         {point_b[2]:.2f}")
        log.info(f"  Disease recall:    {point_b[0]:.4f}")
        log.info(f"  Disease precision: {point_b[1]:.4f}")
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

def plot_sweep(thresholds, recalls, precisions, f1s, points, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, recalls, label='Recall', color='blue')
    plt.plot(thresholds, precisions, label='Precision', color='orange')
    plt.plot(thresholds, f1s, label='F1-macro', color='green')
    
    if points['A']:
        plt.axvline(points['A'][0], color='blue', linestyle='--', label=f"Point A (t={points['A'][0]:.2f})")
    if points['B']:
        plt.axvline(points['B'][2], color='orange', linestyle='--', label=f"Point B (t={points['B'][2]:.2f})")
    if points['C']:
        plt.axvline(points['C'][0], color='green', linestyle='--', label=f"Point C (t={points['C'][0]:.2f})")
        
    plt.xlabel("Threshold")
    plt.ylabel("Metric Score")
    plt.title("Threshold Sweep")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=200)
    plt.close()

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

def predict_with_tta(model, image_pil, n=10):
    preds = []
    for _ in range(n):
        aug = tta_transform(image_pil)
        with torch.no_grad():
            logits = model(aug.unsqueeze(0).to(DEVICE))
            preds.append(torch.softmax(logits, dim=1).cpu())
    return torch.stack(preds).mean(0)  # average over augmentations

def main():
    log.info("Starting fn_reduction pipeline...")
    
    paths, labels = load_data(DATA_ROOT)
    indices = np.arange(len(paths))
    train_idx, test_idx = train_test_split(
        indices,
        train_size=TRAIN_SPLIT,
        stratify=labels,
        random_state=RANDOM_SEED
    )
    
    train_paths, test_paths = paths[train_idx], paths[test_idx]
    train_labels, test_labels = labels[train_idx], labels[test_idx]
    
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    test_dataset = TearDataset(test_paths, test_labels, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = get_model()
    model = model.to(DEVICE)
    
    log.info(f"Loading finetuned checkpoint: {FINETUNED_CKPT}")
    try:
        model.load_state_dict(torch.load(FINETUNED_CKPT, map_location=DEVICE))
    except Exception as e:
        log.error(f"Failed to load finetuned checkpoint: {e}")
        return
        
    model.eval()
    
    # Strategy 1
    log.info("Running Strategy 1 inference...")
    y_true = []
    y_probs_s1 = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Strategy 1 Inference"):
            inputs = inputs.to(DEVICE)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            y_probs_s1.extend(probs[:, 0].cpu().numpy())
            y_true.extend(targets.numpy())
            
    y_true = np.array(y_true)
    y_probs_s1 = np.array(y_probs_s1)
    
    points_s1, thresh_s1, rec_s1, prec_s1, f1_s1 = evaluate_thresholds(y_true, y_probs_s1, "Strategy 1")
    plot_sweep(thresh_s1, rec_s1, prec_s1, f1_s1, points_s1, os.path.join(RESULTS_DIR, "threshold_sweep_s1.png"))
    
    # Strategy 2
    log.info("Running Strategy 2 (TTA) inference...")
    y_probs_s2 = []
    
    for i in tqdm(range(len(test_paths)), desc="Strategy 2 Inference"):
        path = test_paths[i]
        try:
            image_pil = Image.open(path).convert('L')
        except:
            image_pil = Image.fromarray(np.zeros(IMAGE_SIZE, dtype=np.uint8))
        
        avg_probs = predict_with_tta(model, image_pil, n=N_TTA)
        y_probs_s2.append(avg_probs[0, 0].item())
        
    y_probs_s2 = np.array(y_probs_s2)
    points_s2, thresh_s2, rec_s2, prec_s2, f1_s2 = evaluate_thresholds(y_true, y_probs_s2, "Strategy 2 (TTA)")
    plot_sweep(thresh_s2, rec_s2, prec_s2, f1_s2, points_s2, os.path.join(RESULTS_DIR, "threshold_sweep_tta.png"))
    
    # Strategy 3
    log.info("Starting Strategy 3 (Focal Loss Retraining)...")
    
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.RandomAffine(degrees=0, scale=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.2, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    train_dataset = TearDataset(train_paths, train_labels, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)
    
    focal_criterion = FocalLoss(alpha=class_weights, gamma=2.0, reduction='mean')
    
    # Reload original finetuned checkpoint
    model.load_state_dict(torch.load(FINETUNED_CKPT, map_location=DEVICE))
    
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if any(f"features.{i}" in name for i in [6, 7, 8]):
            param.requires_grad = True
        if "classifier" in name:
            param.requires_grad = True
            
    backbone_params = [p for n, p in model.named_parameters()
                       if any(f"features.{i}" in n for i in [6, 7, 8]) and p.requires_grad]
    classifier_params = [p for n, p in model.named_parameters()
                         if "classifier" in n and p.requires_grad]

    optimizer_focal = torch.optim.AdamW([
        {"params": backbone_params,   "lr": 5e-6},
        {"params": classifier_params, "lr": 5e-5},
    ], weight_decay=1e-4)
    
    best_f1_focal = -1.0
    epochs_no_improve_focal = 0
    best_model_wts_focal = copy.deepcopy(model.state_dict())
    
    for epoch in range(1, FOCAL_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in tqdm(train_loader, desc=f"Focal Epoch {epoch:02d}/{FOCAL_EPOCHS} Train", leave=False):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer_focal.zero_grad()
            outputs = model(inputs)
            loss = focal_criterion(outputs, targets)
            loss.backward()
            optimizer_focal.step()
            
            train_loss += loss.item() * inputs.size(0)
            
        train_loss = train_loss / len(train_dataset)
        
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc=f"Focal Epoch {epoch:02d}/{FOCAL_EPOCHS} Val", leave=False):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                
                outputs = model(inputs)
                loss = focal_criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
        val_loss = val_loss / len(test_dataset)
        val_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        
        focal_logger.info(f"Focal Epoch {epoch:02d}/{FOCAL_EPOCHS} | "
                          f"train_loss={train_loss:.4f} | "
                          f"val_loss={val_loss:.4f} | "
                          f"val_f1_macro={val_f1:.4f}")
                 
        if val_f1 > best_f1_focal:
            best_f1_focal = val_f1
            best_model_wts_focal = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), FOCAL_CKPT)
            epochs_no_improve_focal = 0
        else:
            epochs_no_improve_focal += 1
            
        if epochs_no_improve_focal >= FOCAL_PATIENCE:
            focal_logger.info(f"Early stopping at Focal epoch {epoch} — best F1: {best_f1_focal:.4f}")
            break
            
    # Strategy 3 Inference
    log.info("Running Strategy 3 inference...")
    model.load_state_dict(best_model_wts_focal)
    model.eval()
    y_probs_s3 = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Strategy 3 Inference"):
            inputs = inputs.to(DEVICE)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            y_probs_s3.extend(probs[:, 0].cpu().numpy())
            
    y_probs_s3 = np.array(y_probs_s3)
    points_s3, thresh_s3, rec_s3, prec_s3, f1_s3 = evaluate_thresholds(y_true, y_probs_s3, "Strategy 3 (Focal)")
    plot_sweep(thresh_s3, rec_s3, prec_s3, f1_s3, points_s3, os.path.join(RESULTS_DIR, "threshold_sweep_focal.png"))
    
    # Final Comparison
    t_a, r_a, p_a, fn_a, fp_a, f1_a = points_s1['A']
    if points_s1['B']:
        t_b, r_b, p_b, fn_b, fp_b, f1_b = points_s1['B']
    else:
        t_b, r_b, p_b, fn_b, fp_b, f1_b = 0, 0, 0, 0, 0, 0
        
    t_ta, r_ta, p_ta, fn_ta, fp_ta, f1_ta = points_s2['A']
    if points_s2['B']:
        t_tb, r_tb, p_tb, fn_tb, fp_tb, f1_tb = points_s2['B']
    else:
        t_tb, r_tb, p_tb, fn_tb, fp_tb, f1_tb = 0, 0, 0, 0, 0, 0
    t_tc, r_tc, p_tc, fn_tc, fp_tc, f1_tc = points_s2['C']
    
    # Focal best
    s3_preds_05 = np.where(y_probs_s3 >= 0.50, 0, 1)
    r_fl = recall_score(y_true, s3_preds_05, labels=[0, 1], pos_label=0, zero_division=0)
    p_fl = precision_score(y_true, s3_preds_05, labels=[0, 1], pos_label=0, zero_division=0)
    fn_fl = np.sum((y_true == 0) & (s3_preds_05 == 1))
    fp_fl = np.sum((y_true == 1) & (s3_preds_05 == 0))
    
    log.info("=" * 75)
    log.info("FALSE NEGATIVE REDUCTION — FINAL COMPARISON")
    log.info("=" * 75)
    log.info(f"{'Strategy':<30} {'Threshold':>10} {'Recall':>8} {'Precision':>10} {'FN':>5} {'FP':>5}")
    log.info(f"{'Baseline (t=0.50)':<30} {'0.50':>10} {'0.8736':>8} {'0.9744':>10} {'11':>5} {'2':>5}")
    log.info(f"{'S1 Point A (best recall)':<30} {t_a:>10.2f} {r_a:>8.4f} {p_a:>10.4f} {fn_a:>5} {fp_a:>5}")
    log.info(f"{'S1 Point B (recall>=0.95)':<30} {t_b:>10.2f} {r_b:>8.4f} {p_b:>10.4f} {fn_b:>5} {fp_b:>5}")
    log.info(f"{'S2 TTA Point A':<30} {t_ta:>10.2f} {r_ta:>8.4f} {p_ta:>10.4f} {fn_ta:>5} {fp_ta:>5}")
    log.info(f"{'S2 TTA Point B':<30} {t_tb:>10.2f} {r_tb:>8.4f} {p_tb:>10.4f} {fn_tb:>5} {fp_tb:>5}")
    log.info(f"{'S2 TTA Point C':<30} {t_tc:>10.2f} {r_tc:>8.4f} {p_tc:>10.4f} {fn_tc:>5} {fp_tc:>5}")
    log.info(f"{'S3 Focal Loss best':<30} {'0.50':>10} {r_fl:>8.4f} {p_fl:>10.4f} {fn_fl:>5} {fp_fl:>5}")

if __name__ == "__main__":
    main()
