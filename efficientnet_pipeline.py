#!/usr/bin/env python3
"""Transfer Learning pipeline using EfficientNet-B0 for Tear Microscopy."""

import os
import logging
import copy
from datetime import datetime
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

# =============================================================================
# Config
# =============================================================================
DATA_ROOT    = "/home/rootofpower/personal/hackkosice2026/ml_model/test"
RESULTS_DIR  = "results"
LOG_FILE     = "results/efficientnet_run.log"
IMAGE_SIZE   = (490, 490)
BATCH_SIZE   = 16
NUM_EPOCHS   = 60
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 12        # early stopping
TRAIN_SPLIT  = 0.80
RANDOM_SEED  = 42
NUM_CLASSES  = 2
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Logging
# =============================================================================
os.makedirs(RESULTS_DIR, exist_ok=True)

log_path = os.path.join(RESULTS_DIR, "efficientnet_run.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()

# =============================================================================
# Dataset Definition
# =============================================================================
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
            # Fallback to a zero image if unreadable
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

# =============================================================================
# Main Script
# =============================================================================
def main():
    log.info("Starting EfficientNet Transfer Learning Pipeline")
    
    paths, labels = load_data(DATA_ROOT)
    
    if len(paths) == 0:
        log.info("No images found. Exiting.")
        return
        
    log.info(f"Total images found: {len(paths)}")
    label_counts = Counter(labels)
    log.info(f"Class 0 (diabetes_spolu_zo_suchym_okom): {label_counts[0]}")
    log.info(f"Class 1 (zdravi_ludi): {label_counts[1]}")
    
    # Split
    indices = np.arange(len(paths))
    train_idx, test_idx = train_test_split(
        indices,
        train_size=TRAIN_SPLIT,
        stratify=labels,
        random_state=RANDOM_SEED
    )
    
    train_paths, test_paths = paths[train_idx], paths[test_idx]
    train_labels, test_labels = labels[train_idx], labels[test_idx]
    
    log.info(f"Training samples: {len(train_paths)}")
    log.info(f"Test samples: {len(test_paths)}")
    
    # Transforms
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
    
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    train_dataset = TearDataset(train_paths, train_labels, transform=train_transform)
    test_dataset = TearDataset(test_paths, test_labels, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model Setup
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier head only
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(1280, NUM_CLASSES)
    )
    model = model.to(DEVICE)

    log.info(f"Device: {DEVICE}")
    log.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss and Optimizer
    weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.classifier.parameters(),
                                   lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Training Loop
    best_f1 = -1.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    
    best_checkpoint_path = os.path.join(RESULTS_DIR, "efficientnet_best.pth")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{NUM_EPOCHS} Train", leave=False):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
        train_loss = train_loss / len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc=f"Epoch {epoch:02d}/{NUM_EPOCHS} Val", leave=False):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
        val_loss = val_loss / len(test_dataset)
        val_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        
        log.info(f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
                 f"train_loss={train_loss:.4f} | "
                 f"val_loss={val_loss:.4f} | "
                 f"val_f1_macro={val_f1:.4f} | "
                 f"lr={scheduler.get_last_lr()[0]:.6f}")
                 
        scheduler.step()
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), best_checkpoint_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= PATIENCE:
            log.info(f"Early stopping at epoch {epoch} — best F1: {best_f1:.4f}")
            break
            
    # Evaluation (on test set, using best checkpoint)
    log.info("Loading best model for final evaluation on test set...")
    model.load_state_dict(best_model_wts)
    model.eval()
    
    y_true = []
    y_pred = []
    y_proba = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_proba.extend(probs[:, 1].cpu().numpy())  # Probs for class 1
            
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    y_proba_class0_np = 1.0 - np.array(y_proba) # Prob of class 0
    
    try:
        positive_mask = (y_true_np == 0).astype(np.int64)
        roc_auc = roc_auc_score(positive_mask, y_proba_class0_np)
    except Exception as e:
        log.info(f"Could not compute ROC-AUC: {e}")
        roc_auc = 0.0
        
    f1_macro = f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    
    # Disease recall = recall for class 0
    recall = recall_score(y_true_np, y_pred_np, labels=[0, 1], pos_label=0, zero_division=0)
    accuracy = accuracy_score(y_true_np, y_pred_np)
    
    class_names = ["diabetes_spolu_zo_suchym_okom", "zdravi_ludi"]
    
    log.info("=" * 60)
    log.info("EFFICIENTNET RESULTS")
    log.info("=" * 60)
    log.info(f"ROC-AUC:        {roc_auc:.4f}")
    log.info(f"F1-macro:       {f1_macro:.4f}")
    log.info(f"Disease recall: {recall:.4f}")
    log.info(f"Accuracy:       {accuracy:.4f}")
    log.info("\nClassification report:")
    log.info("\n" + classification_report(y_true_np, y_pred_np, target_names=class_names, zero_division=0))
    
    # Save confusion matrix PNG
    cm = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predikovane")
    ax.set_ylabel("Skutocne")
    ax.set_title("EfficientNet — Confusion Matrix")
    fig.tight_layout()
    confusion_path = os.path.join(RESULTS_DIR, "efficientnet_confusion_matrix.png")
    fig.savefig(confusion_path, dpi=200)
    plt.close(fig)
    
    # Final comparison block at end of log
    log.info("=" * 60)
    log.info("COMPARISON — RF baseline vs EfficientNet")
    log.info("=" * 60)
    log.info(f"{'Metric':<25} {'RF baseline':>15} {'EfficientNet':>15}")
    log.info(f"{'ROC-AUC':<25} {'0.8548':>15} {roc_auc:>15.4f}")
    log.info(f"{'F1-macro':<25} {'0.6867':>15} {f1_macro:>15.4f}")
    log.info(f"{'Disease recall':<25} {'0.4598':>15} {recall:>15.4f}")
    log.info(f"Log saved to: {log_path}")

if __name__ == "__main__":
    main()
