#!/usr/bin/env python3
"""EfficientNet v2 pipeline — improved 2-phase training with Mixup, Focal Loss,
   label smoothing, deeper unfreezing, and optional B2 backbone.

   Phase 1: Frozen backbone → train classifier head (warm-up)
   Phase 2: Unfreeze features.4-8 → Focal Loss + Mixup + CosineWarmRestarts
"""

import os
import argparse
import logging
import copy
from pathlib import Path

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
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from config import PAIRS

# =============================================================================
# Constants
# =============================================================================
SUPPORTED_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# =============================================================================
# Dataset
# =============================================================================
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
            image = Image.open(path).convert("L")
        except Exception:
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
        log.info(
            f"{split_name} — disease: {labels.count(0)}, "
            f"healthy: {labels.count(1)}, "
            f"total: {len(labels)}"
        )
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
                f"[{pair_key}] Path not found: {path}\n  Key: {key}"
            )
        files = [
            f
            for f in p.glob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if len(files) == 0:
            raise ValueError(
                f"[{pair_key}] No images found in: {path}\n  Key: {key}"
            )
        log.info(f"  {key}: {len(files)} images → {path}")


# =============================================================================
# Focal Loss
# =============================================================================
class FocalLoss(nn.Module):
    """Focal Loss with per-class alpha weights and configurable gamma."""

    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss
        return focal.mean() if self.reduction == "mean" else focal


# =============================================================================
# Mixup
# =============================================================================
def mixup_data(x, y, alpha=0.2):
    """Mixup: blend pairs of images and labels."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for mixup samples."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =============================================================================
# Model Builder
# =============================================================================
def build_model(backbone: str, num_classes: int = 2, pretrained: bool = True):
    """Build EfficientNet model with specified backbone."""
    if backbone == "b0":
        weights = "IMAGENET1K_V1" if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = 1280
    elif backbone == "b2":
        weights = "IMAGENET1K_V1" if pretrained else None
        model = models.efficientnet_b2(weights=weights)
        in_features = 1408
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


# =============================================================================
# Training Helpers
# =============================================================================
def train_one_epoch(
    model, loader, criterion, optimizer, device, dataset_len, use_mixup=False, log=None
):
    model.train()
    running_loss = 0.0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        if use_mixup:
            inputs, y_a, y_b, lam = mixup_data(inputs, targets, alpha=0.2)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        else:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / dataset_len


def evaluate(model, loader, criterion, device, dataset_len):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    val_loss = running_loss / dataset_len
    val_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return val_loss, val_f1, np.array(all_preds), np.array(all_targets)


def final_evaluation(model, loader, device, pair_label, log):
    """Full evaluation with all metrics."""
    model.eval()
    y_true, y_pred, y_proba = [], [], []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
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
    except Exception:
        roc_auc = 0.0

    f1_macro = f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)
    rec = recall_score(y_true_np, y_pred_np, pos_label=0, zero_division=0)
    prec = precision_score(y_true_np, y_pred_np, pos_label=0, zero_division=0)
    fn = int(np.sum((y_true_np == 0) & (y_pred_np == 1)))
    fp = int(np.sum((y_true_np == 1) & (y_pred_np == 0)))

    class_names = [pair_label.split()[0], "Zdravi"]

    log.info("=" * 65)
    log.info("FINAL EVALUATION")
    log.info("=" * 65)
    log.info(f"ROC-AUC:           {roc_auc:.4f}")
    log.info(f"F1-macro:          {f1_macro:.4f}")
    log.info(f"Disease recall:    {rec:.4f}")
    log.info(f"Disease precision: {prec:.4f}")
    log.info(f"False negatives:   {fn}")
    log.info(f"False positives:   {fp}")
    log.info(
        "\nClassification report:\n"
        + classification_report(
            y_true_np, y_pred_np, target_names=class_names, zero_division=0
        )
    )

    return y_true_np, y_pred_np, class_names, f1_macro, fn, fp


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="EfficientNet v2 — improved 2-phase training"
    )
    parser.add_argument("--pair", required=True, choices=list(PAIRS.keys()))
    parser.add_argument(
        "--backbone",
        default="b2",
        choices=["b0", "b2"],
        help="EfficientNet backbone variant (default: b2)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    PAIR_KEY = args.pair
    PAIR = PAIRS[PAIR_KEY]
    BACKBONE = args.backbone
    SEED = args.seed

    RESULTS_DIR = f"results/{PAIR_KEY}"
    BATCH_SIZE = 16
    NUM_CLASSES = 2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Phase 1 config
    PHASE1_EPOCHS = 15
    PHASE1_LR = 1e-3
    PHASE1_PATIENCE = 8

    # Phase 2 config
    PHASE2_EPOCHS = 60
    PHASE2_PATIENCE = 15
    PHASE2_BACKBONE_LR = 5e-6
    PHASE2_HEAD_LR = 5e-5
    FOCAL_GAMMA = 3.0
    LABEL_SMOOTHING = 0.05
    DISEASE_WEIGHT_BOOST = 1.5  # extra boost on disease class weight
    USE_MIXUP = True

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    seed_tag = f"_s{SEED}" if SEED != 42 else ""
    CHECKPOINT = f"{RESULTS_DIR}/efficientnet_v2_{BACKBONE}{seed_tag}_{PAIR_KEY}.pth"
    LOG_FILE = f"{RESULTS_DIR}/training_v2_{BACKBONE}{seed_tag}_{PAIR_KEY}.log"

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Configure logging — clear old handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger()

    log.info(f"Starting EfficientNet v2 Training for pair: {PAIR_KEY}")
    log.info(f"Label: {PAIR['label']}")
    log.info(f"Backbone: {BACKBONE} | Seed: {SEED}")
    log.info(f"Device: {DEVICE}")

    validate_paths(PAIR_KEY, log)
    train_paths, train_labels, test_paths, test_labels = load_data(PAIR, log)

    if len(train_paths) == 0:
        log.info("No training images found. Exiting.")
        return

    # ---- Transforms ----
    train_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.RandomAffine(degrees=0, scale=(0.85, 1.15)),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5)),
            transforms.ColorJitter(brightness=0.3, contrast=0.4),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = TearDataset(train_paths, train_labels, transform=train_transform)
    test_dataset = TearDataset(test_paths, test_labels, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    # ---- Compute class weights ----
    classes = np.unique(train_labels)
    if len(classes) > 1:
        weights = compute_class_weight("balanced", classes=classes, y=train_labels)
        # Boost disease class weight
        weights[0] *= DISEASE_WEIGHT_BOOST
        class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)
        log.info(f"Class weights (disease-boosted): {weights}")
    else:
        class_weights = None

    # ---- Build model ----
    model = build_model(BACKBONE, NUM_CLASSES, pretrained=True)
    model = model.to(DEVICE)

    # ================================================================
    # PHASE 1: Frozen backbone — train classifier head only
    # ================================================================
    log.info("=" * 65)
    log.info("PHASE 1: Frozen backbone — classifier head warm-up")
    log.info("=" * 65)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable_p1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Phase 1 trainable params: {trainable_p1:,}")

    criterion_p1 = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    optimizer_p1 = torch.optim.AdamW(
        model.classifier.parameters(), lr=PHASE1_LR, weight_decay=1e-4
    )
    scheduler_p1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p1, T_max=PHASE1_EPOCHS
    )

    best_f1_p1 = -1.0
    best_wts_p1 = copy.deepcopy(model.state_dict())
    no_improve_p1 = 0

    for epoch in range(1, PHASE1_EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion_p1, optimizer_p1, DEVICE, len(train_dataset),
            use_mixup=False, log=log,
        )
        val_loss, val_f1, _, _ = evaluate(
            model, test_loader, criterion_p1, DEVICE, len(test_dataset)
        )

        log.info(
            f"P1 Epoch {epoch:02d}/{PHASE1_EPOCHS} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_f1={val_f1:.4f} | lr={optimizer_p1.param_groups[0]['lr']:.2e}"
        )

        scheduler_p1.step()

        if val_f1 > best_f1_p1:
            best_f1_p1 = val_f1
            best_wts_p1 = copy.deepcopy(model.state_dict())
            no_improve_p1 = 0
        else:
            no_improve_p1 += 1

        if no_improve_p1 >= PHASE1_PATIENCE:
            log.info(f"Phase 1 early stop at epoch {epoch} — best F1: {best_f1_p1:.4f}")
            break

    model.load_state_dict(best_wts_p1)
    log.info(f"Phase 1 complete — best F1: {best_f1_p1:.4f}")

    # ================================================================
    # PHASE 2: Unfreeze deep layers — Focal Loss + Mixup
    # ================================================================
    log.info("=" * 65)
    log.info("PHASE 2: Deep fine-tune with Focal Loss + Mixup")
    log.info("=" * 65)

    # Unfreeze features.4 through features.8 + classifier
    for name, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if any(f"features.{i}" in name for i in [4, 5, 6, 7, 8]):
            param.requires_grad = True
        if "classifier" in name:
            param.requires_grad = True

    trainable_p2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Phase 2 trainable params: {trainable_p2:,}")

    focal_criterion = FocalLoss(
        alpha=class_weights,
        gamma=FOCAL_GAMMA,
        label_smoothing=LABEL_SMOOTHING,
        reduction="mean",
    )

    backbone_params = [
        p
        for n, p in model.named_parameters()
        if any(f"features.{i}" in n for i in [4, 5, 6, 7, 8]) and p.requires_grad
    ]
    classifier_params = [
        p
        for n, p in model.named_parameters()
        if "classifier" in n and p.requires_grad
    ]

    optimizer_p2 = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": PHASE2_BACKBONE_LR},
            {"params": classifier_params, "lr": PHASE2_HEAD_LR},
        ],
        weight_decay=1e-4,
    )

    # CosineAnnealingWarmRestarts for better exploration
    scheduler_p2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_p2, T_0=10, T_mult=2
    )

    best_f1_p2 = -1.0
    best_wts_p2 = copy.deepcopy(model.state_dict())
    no_improve_p2 = 0

    for epoch in range(1, PHASE2_EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, focal_criterion, optimizer_p2, DEVICE, len(train_dataset),
            use_mixup=USE_MIXUP, log=log,
        )
        val_loss, val_f1, _, _ = evaluate(
            model, test_loader, focal_criterion, DEVICE, len(test_dataset)
        )

        log.info(
            f"P2 Epoch {epoch:02d}/{PHASE2_EPOCHS} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_f1={val_f1:.4f} | "
            f"bb_lr={optimizer_p2.param_groups[0]['lr']:.2e} | "
            f"head_lr={optimizer_p2.param_groups[1]['lr']:.2e}"
        )

        scheduler_p2.step()

        if val_f1 > best_f1_p2:
            best_f1_p2 = val_f1
            best_wts_p2 = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), CHECKPOINT)
            no_improve_p2 = 0
        else:
            no_improve_p2 += 1

        if no_improve_p2 >= PHASE2_PATIENCE:
            log.info(f"Phase 2 early stop at epoch {epoch} — best F1: {best_f1_p2:.4f}")
            break

    # ================================================================
    # Final Evaluation
    # ================================================================
    log.info("Loading best Phase 2 model for final evaluation...")
    model.load_state_dict(best_wts_p2)

    y_true_np, y_pred_np, class_names, f1_macro, fn, fp = final_evaluation(
        model, test_loader, DEVICE, PAIR["label"], log
    )

    # Save confusion matrix
    cm = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predikovane")
    ax.set_ylabel("Skutocne")
    ax.set_title(f"EfficientNet v2 {BACKBONE} ({PAIR_KEY})")
    fig.tight_layout()
    cm_path = os.path.join(
        RESULTS_DIR, f"v2_{BACKBONE}{seed_tag}_confusion_matrix_{PAIR_KEY}.png"
    )
    fig.savefig(cm_path, dpi=200)
    plt.close(fig)

    log.info(f"Model saved to: {CHECKPOINT}")
    log.info(f"Log saved to: {LOG_FILE}")
    log.info(f"Confusion matrix: {cm_path}")


if __name__ == "__main__":
    main()
