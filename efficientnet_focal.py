#!/usr/bin/env python3
"""Transfer Learning pipeline using EfficientNet-B0 (Focal Loss Retraining)."""

import os
import argparse
import logging
import copy
from datetime import datetime
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from config import PAIRS

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
            image = Image.fromarray(np.zeros((490, 490), dtype=np.uint8))
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def load_data(pair_config: dict, log):
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

def validate_paths(pair_key: str, log):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True, choices=list(PAIRS.keys()))
    args = parser.parse_args()
    
    PAIR_KEY = args.pair
    PAIR = PAIRS[PAIR_KEY]
    
    RESULTS_DIR  = f"results/{PAIR_KEY}"
    IMAGE_SIZE   = (490, 490)
    BATCH_SIZE   = 16
    FOCAL_EPOCHS   = 20
    FOCAL_PATIENCE = 8
    RANDOM_SEED  = 42
    NUM_CLASSES  = 2
    DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    CHECKPOINT_FINETUNED = f"{RESULTS_DIR}/efficientnet_finetuned_{PAIR_KEY}.pth"
    CHECKPOINT_FOCAL     = f"{RESULTS_DIR}/efficientnet_focal_{PAIR_KEY}.pth"
    LOG_FILE_FOCAL       = f"{RESULTS_DIR}/efficientnet_focal_{PAIR_KEY}.log"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(LOG_FILE_FOCAL, mode="w", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    log = logging.getLogger()
    
    log.info(f"Starting EfficientNet Focal Retraining for pair: {PAIR_KEY}")
    log.info(f"Label: {PAIR['label']}")
    
    validate_paths(PAIR_KEY, log)
    
    train_paths, train_labels, test_paths, test_labels = load_data(PAIR, log)
    
    if len(train_paths) == 0:
        log.info("No training images found. Exiting.")
        return
    
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
    
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(1280, NUM_CLASSES)
    )
    model = model.to(DEVICE)
    
    log.info(f"Loading finetuned checkpoint from {CHECKPOINT_FINETUNED}...")
    try:
        model.load_state_dict(torch.load(CHECKPOINT_FINETUNED, map_location=DEVICE))
    except Exception as e:
        log.error(f"Failed to load checkpoint: {e}. You must run efficientnet_finetune.py first.")
        return
        
    classes = np.unique(train_labels)
    if len(classes) > 1:
        weights = compute_class_weight('balanced', classes=classes, y=train_labels)
        class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)
    else:
        class_weights = None
        
    focal_criterion = FocalLoss(alpha=class_weights, gamma=2.0, reduction='mean')
    
    # Unfreeze last 3 blocks + classifier
    for name, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if any(f"features.{i}" in name for i in [6, 7, 8]):
            param.requires_grad = True
        if "classifier" in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Focal retrain trainable params: {trainable:,}")

    backbone_params = [p for n, p in model.named_parameters()
                       if any(f"features.{i}" in n for i in [6, 7, 8])
                       and p.requires_grad]
    classifier_params = [p for n, p in model.named_parameters()
                         if "classifier" in n and p.requires_grad]

    optimizer_focal = torch.optim.AdamW([
        {"params": backbone_params,   "lr": 5e-6},
        {"params": classifier_params, "lr": 5e-5},
    ], weight_decay=1e-4)

    best_f1_focal = -1.0
    best_model_wts_focal = copy.deepcopy(model.state_dict())
    epochs_no_improve_focal = 0

    for epoch in range(1, FOCAL_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{FOCAL_EPOCHS} Train", leave=False):
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
            for inputs, targets in tqdm(test_loader, desc=f"Epoch {epoch:02d}/{FOCAL_EPOCHS} Val", leave=False):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                
                outputs = model(inputs)
                loss = focal_criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
        val_loss = val_loss / len(test_dataset)
        val_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        
        log.info(f"Epoch {epoch:02d}/{FOCAL_EPOCHS} | "
                 f"train_loss={train_loss:.4f} | "
                 f"val_loss={val_loss:.4f} | "
                 f"val_f1_macro={val_f1:.4f} | "
                 f"backbone_lr={optimizer_focal.param_groups[0]['lr']:.2e} | "
                 f"head_lr={optimizer_focal.param_groups[1]['lr']:.2e}")
                 
        if val_f1 > best_f1_focal:
            best_f1_focal = val_f1
            best_model_wts_focal = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), CHECKPOINT_FOCAL)
            epochs_no_improve_focal = 0
        else:
            epochs_no_improve_focal += 1
            
        if epochs_no_improve_focal >= FOCAL_PATIENCE:
            log.info(f"Early stopping at epoch {epoch} — best F1: {best_f1_focal:.4f}")
            break
            
    log.info("Loading best focal model for final evaluation on test set...")
    model.load_state_dict(best_model_wts_focal)
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
            y_proba.extend(probs[:, 1].cpu().numpy())
            
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    y_proba_class0_np = 1.0 - np.array(y_proba)
    
    try:
        positive_mask = (y_true_np == 0).astype(np.int64)
        roc_auc = roc_auc_score(positive_mask, y_proba_class0_np)
    except Exception as e:
        log.info(f"Could not compute ROC-AUC: {e}")
        roc_auc = 0.0
        
    f1_macro = f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
    recall = recall_score(y_true_np, y_pred_np, labels=[0, 1], pos_label=0, zero_division=0)
    precision = precision_score(y_true_np, y_pred_np, labels=[0, 1], pos_label=0, zero_division=0)
    fn = sum((y_true_np == 0) & (y_pred_np == 1))
    
    class_names = [PAIR["label"].split()[0], "Zdravi"]
    
    log.info("=" * 65)
    log.info("FOCAL RETRAINING EVALUATION")
    log.info("=" * 65)
    log.info(f"ROC-AUC:           {roc_auc:.4f}")
    log.info(f"F1-macro:          {f1_macro:.4f}")
    log.info(f"Disease recall:    {recall:.4f}")
    log.info(f"Disease precision: {precision:.4f}")
    log.info(f"False negatives:   {fn}")
    
    log.info("\nClassification report:")
    log.info("\n" + classification_report(y_true_np, y_pred_np, target_names=class_names, zero_division=0))
    
    cm = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predikovane")
    ax.set_ylabel("Skutocne")
    ax.set_title(f"EfficientNet Focal ({PAIR_KEY})")
    fig.tight_layout()
    confusion_path = os.path.join(RESULTS_DIR, f"efficientnet_focal_confusion_matrix_{PAIR_KEY}.png")
    fig.savefig(confusion_path, dpi=200)
    plt.close(fig)
    
    log.info(f"Focal model successfully trained and saved to {CHECKPOINT_FOCAL}")
    log.info(f"Log saved to: {LOG_FILE_FOCAL}")

if __name__ == "__main__":
    main()
