#!/usr/bin/env python3
"""Transfer Learning pipeline using EfficientNet-B0 (Frozen Training)."""

import os
import argparse
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

from sklearn.metrics import f1_score
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
            # Fallback to a zero image if unreadable
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True, choices=list(PAIRS.keys()))
    args = parser.parse_args()
    
    PAIR_KEY = args.pair
    PAIR = PAIRS[PAIR_KEY]
    
    RESULTS_DIR  = f"results/{PAIR_KEY}"
    IMAGE_SIZE   = (490, 490)
    BATCH_SIZE   = 16
    NUM_EPOCHS   = 60
    LR           = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE     = 12
    RANDOM_SEED  = 42
    NUM_CLASSES  = 2
    DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    CHECKPOINT_FROZEN = f"{RESULTS_DIR}/efficientnet_frozen_{PAIR_KEY}.pth"
    LOG_FILE_FROZEN   = f"{RESULTS_DIR}/training_frozen_{PAIR_KEY}.log"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(LOG_FILE_FROZEN, mode="w", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    log = logging.getLogger()
    
    log.info(f"Starting EfficientNet Frozen Training for pair: {PAIR_KEY}")
    log.info(f"Label: {PAIR['label']}")
    
    validate_paths(PAIR_KEY, log)
    
    train_paths, train_labels, test_paths, test_labels = load_data(PAIR, log)
    
    if len(train_paths) == 0:
        log.info("No training images found. Exiting.")
        return
        
    log.info(f"Training samples: {len(train_paths)}")
    log.info(f"Test samples: {len(test_paths)}")
    
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
    
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(1280, NUM_CLASSES)
    )
    model = model.to(DEVICE)

    log.info(f"Device: {DEVICE}")
    log.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    classes = np.unique(train_labels)
    if len(classes) > 1:
        weights = compute_class_weight('balanced', classes=classes, y=train_labels)
        class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)
    else:
        class_weights = None
        
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.classifier.parameters(),
                                   lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    best_f1 = -1.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    
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
                 f"lr={optimizer.param_groups[0]['lr']:.2e}")
                 
        scheduler.step()
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), CHECKPOINT_FROZEN)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= PATIENCE:
            log.info(f"Early stopping at epoch {epoch} — best F1: {best_f1:.4f}")
            break

    log.info(f"Finished. Best model saved to: {CHECKPOINT_FROZEN}")

if __name__ == "__main__":
    main()
