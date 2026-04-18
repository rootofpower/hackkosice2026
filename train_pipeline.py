#!/usr/bin/env python3
"""Phase 0 binary classification pipeline for tear microscopy images."""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import cv2
import matplotlib
matplotlib.use("Agg")
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

# Config
IMAGE_SIZE = (523, 490)
TRAIN_SPLIT = 0.80
TEST_SPLIT = 0.20
RANDOM_SEED = 42
ALPHA_DATA_ROOT = Path("alpha_model")
BETA_DATA_ROOT = Path("beta_model")
RESULTS_DIR = Path("results")

SUPPORTED_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

LBP_POINTS = 16
LBP_RADIUS = 2
LBP_METHOD = "uniform"

GLCM_DISTANCES = [1, 2]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_PROPERTIES = ("contrast", "energy", "homogeneity", "correlation")

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)


@dataclass(frozen=True)
class ClassSpec:
    label: int
    folder_name: str
    display_name: str
    filename_prefix: str | None = None


PHASE0_CLASSES = (
    ClassSpec(label=0, folder_name="chore_suche", display_name="Suche Oko"),
    ClassSpec(label=1, folder_name="zdrave", display_name="Zdravi Ludia"),
)

BETA_PHASE0_CLASSES = (
    ClassSpec(label=0, folder_name="skleroza", display_name="Skleroza Multiplex"),
    ClassSpec(label=1, folder_name="zdravi", display_name="Zdravi Ludia"),
)

GAMMA_DATA_ROOT = Path("gama_model")
GAMMA_PHASE0_CLASSES = (
    ClassSpec(
        label=0,
        folder_name="chore_diabetes",
        display_name="Diabetes",
        filename_prefix="diabetes_",
    ),
    ClassSpec(
        label=1,
        folder_name="chore_a_zdravi",
        display_name="Ostatne Triedy",
    ),
)

PHASE0_DATASET_PROFILES = {
    "alpha": {
        "data_root": ALPHA_DATA_ROOT,
        "class_specs": PHASE0_CLASSES,
        "result_prefix": "alpha_phase0",
        "display_name": "alpha_model",
    },
    "beta": {
        "data_root": BETA_DATA_ROOT,
        "class_specs": BETA_PHASE0_CLASSES,
        "result_prefix": "beta_phase0",
        "display_name": "beta_model",
    },
    "gamma": {
        "data_root": GAMMA_DATA_ROOT,
        "class_specs": GAMMA_PHASE0_CLASSES,
        "result_prefix": "gamma_phase0",
        "display_name": "gamma_model",
    },
}


def expected_image_shape() -> tuple[int, int]:
    """OpenCV loads grayscale images as (height, width)."""
    return IMAGE_SIZE[1], IMAGE_SIZE[0]


def collect_image_paths(data_root: Path, class_specs: tuple[ClassSpec, ...]) -> list[tuple[Path, ClassSpec]]:
    samples: list[tuple[Path, ClassSpec]] = []

    for class_spec in class_specs:
        class_dir = data_root / class_spec.folder_name
        if not class_dir.exists():
            warnings.warn(f"Missing class directory: {class_dir}")
            continue

        image_paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
            and (
                class_spec.filename_prefix is None
                or path.name.lower().startswith(class_spec.filename_prefix.lower())
            )
        )
        if not image_paths:
            warnings.warn(f"No images found for class '{class_spec.display_name}' in {class_dir}")

        samples.extend((path, class_spec) for path in image_paths)

    return samples


def load_grayscale_image(image_path: Path) -> np.ndarray | None:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        warnings.warn(f"Skipping unreadable image: {image_path}")
        return None

    if image.ndim != 2:
        warnings.warn(f"Skipping non-grayscale image: {image_path}")
        return None

    if image.shape != expected_image_shape():
        warnings.warn(
            f"Skipping image with unexpected shape {image.shape}: {image_path}. Expected {expected_image_shape()}."
        )
        return None

    return image


def load_dataset(data_root: Path, class_specs: tuple[ClassSpec, ...]) -> tuple[list[np.ndarray], np.ndarray, list[Path]]:
    samples = collect_image_paths(data_root, class_specs)
    if not samples:
        raise RuntimeError(f"No readable image candidates found under {data_root.resolve()}.")

    images: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[Path] = []

    for image_path, class_spec in tqdm(samples, desc="Loading Phase 0 images", unit="image"):
        image = load_grayscale_image(image_path)
        if image is None:
            continue
        images.append(image)
        labels.append(class_spec.label)
        paths.append(image_path)

    label_counts = Counter(labels)
    missing_labels = [spec.display_name for spec in class_specs if label_counts.get(spec.label, 0) == 0]
    if missing_labels:
        raise RuntimeError(f"Phase 0 cannot run because these classes have no valid images: {', '.join(missing_labels)}")

    return images, np.asarray(labels, dtype=np.int64), paths


def extract_lbp_features(image: np.ndarray) -> np.ndarray:
    lbp = local_binary_pattern(image, P=LBP_POINTS, R=LBP_RADIUS, method=LBP_METHOD)
    bins = np.arange(0, LBP_POINTS + 3)
    histogram, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, LBP_POINTS + 2))
    histogram = histogram.astype(np.float32)
    histogram /= histogram.sum() + 1e-12
    return histogram


def extract_glcm_features(image: np.ndarray) -> np.ndarray:
    glcm = graycomatrix(
        image,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=256,
        symmetric=True,
        normed=True,
    )
    features = [graycoprops(glcm, prop).ravel() for prop in GLCM_PROPERTIES]
    return np.concatenate(features).astype(np.float32)


def extract_hog_features(image: np.ndarray) -> np.ndarray:
    features = hog(
        image,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
        channel_axis=None,
    )
    return features.astype(np.float32)


def extract_shape_features(image: np.ndarray) -> np.ndarray:
    _, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contour_result = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contour_result[0] if len(contour_result) == 2 else contour_result[1]

    if not contours:
        return np.zeros(4, dtype=np.float32)

    largest_contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest_contour))
    perimeter = float(cv2.arcLength(largest_contour, closed=True))
    circularity = (4.0 * math.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

    hull = cv2.convexHull(largest_contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = (area / hull_area) if hull_area > 0 else 0.0

    return np.asarray([area, perimeter, circularity, solidity], dtype=np.float32)


def extract_feature_vector(image: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            extract_lbp_features(image),
            extract_glcm_features(image),
            extract_hog_features(image),
            extract_shape_features(image),
        ]
    ).astype(np.float32)


def extract_feature_matrix(images: list[np.ndarray]) -> np.ndarray:
    feature_rows = [
        extract_feature_vector(image) for image in tqdm(images, desc="Extracting features", unit="image")
    ]
    feature_lengths = {row.shape[0] for row in feature_rows}
    if len(feature_lengths) != 1:
        raise RuntimeError(f"Feature extraction produced inconsistent vector lengths: {sorted(feature_lengths)}")
    return np.vstack(feature_rows)


def train_classifier(x_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes.tolist(), weights.tolist()))
    pca_components = min(50, x_train.shape[0], x_train.shape[1])

    classifier = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_components, random_state=RANDOM_SEED)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight=class_weight_dict,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    classifier.fit(x_train, y_train)
    return classifier


def evaluate_classifier(
    classifier: Pipeline,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_specs: tuple[ClassSpec, ...],
) -> dict[str, Any]:
    ordered_labels = [spec.label for spec in class_specs]
    display_names = [spec.display_name for spec in class_specs]

    train_diabetes_prob = classifier.predict_proba(x_train)[:, 0]
    fpr, tpr, thresholds = roc_curve(y_train, train_diabetes_prob, pos_label=0)
    j_scores = tpr - fpr
    best_threshold = float(thresholds[int(np.argmax(j_scores))])

    y_proba = classifier.predict_proba(x_test)[:, 0]
    y_pred = np.where(y_proba >= best_threshold, 0, 1)
    diabetes_probs = y_proba[y_test == 0]

    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=ordered_labels,
        zero_division=0,
    )
    report = classification_report(
        y_test,
        y_pred,
        labels=ordered_labels,
        target_names=display_names,
        zero_division=0,
    )

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "roc_auc": None,
        "optimal_threshold": best_threshold,
        "diabetes_predicted_probs": diabetes_probs.round(3).tolist(),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=ordered_labels).tolist(),
        "class_names": display_names,
        "classification_report": report,
        "per_class": {},
        "macro_avg": {
            "precision": float(np.mean(precision)),
            "recall": float(np.mean(recall)),
            "f1": float(np.mean(per_class_f1)),
        },
    }

    try:
        positive_mask = (y_test == ordered_labels[0]).astype(np.int64)
        metrics["roc_auc"] = float(roc_auc_score(positive_mask, y_proba))
    except ValueError as error:
        warnings.warn(f"ROC-AUC could not be computed: {error}")

    for index, class_spec in enumerate(class_specs):
        metrics["per_class"][class_spec.display_name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(per_class_f1[index]),
            "support": int(support[index]),
        }

    return metrics


def plot_confusion_matrix(matrix: np.ndarray, class_specs: tuple[ClassSpec, ...], output_path: Path) -> None:
    display_names = [spec.display_name for spec in class_specs]

    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=display_names,
        yticklabels=display_names,
        ax=axis,
    )
    axis.set_xlabel("Predikovane")
    axis.set_ylabel("Skutocne")
    axis.set_title("Phase 0 Confusion Matrix")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)


def print_phase0_results(metrics: dict[str, Any], metrics_path: Path, confusion_path: Path) -> None:
    print("Phase 0 results:")
    print(f"  Dataset profile: {metrics['dataset']['profile']}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1-macro: {metrics['f1_macro']:.4f}")
    if metrics["roc_auc"] is None:
        print("  ROC-AUC: unavailable")
    else:
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  Optimal threshold: {metrics['optimal_threshold']:.3f}")
    print(f"  Diabetes predicted probs: {np.asarray(metrics['diabetes_predicted_probs']).round(3)}")
    print(f"  PCA components selected: {metrics['pca_components_selected']}")
    print(
        f"  PCA kept {metrics['pca_components_selected']} components explaining "
        f"{metrics['pca_variance_explained']:.1%} of variance"
    )
    print("  Confusion matrix:")
    for row in metrics["confusion_matrix"]:
        print(f"    {row}")
    print("  Classification report:")
    print(metrics["classification_report"])
    print(f"  Metrics JSON: {metrics_path}")
    print(f"  Confusion matrix PNG: {confusion_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 0 tear microscopy classification pipeline.")
    parser.add_argument(
        "--dataset-profile",
        choices=sorted(PHASE0_DATASET_PROFILES),
        default="alpha",
        help="Select which Phase 0 dataset profile to use.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_profile = PHASE0_DATASET_PROFILES[args.dataset_profile]
    data_root: Path = dataset_profile["data_root"]
    class_specs: tuple[ClassSpec, ...] = dataset_profile["class_specs"]
    result_prefix: str = dataset_profile["result_prefix"]
    display_name: str = dataset_profile["display_name"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {data_root.resolve()}")
    images, labels, paths = load_dataset(data_root, class_specs)

    label_counts = Counter(labels.tolist())
    print("Discovered valid images per class:")
    for class_spec in class_specs:
        print(f"  {class_spec.display_name}: {label_counts[class_spec.label]}")

    features = extract_feature_matrix(images)
    print(f"Feature matrix shape: {features.shape}")

    split_indices = np.arange(len(paths))
    train_idx, test_idx = train_test_split(
        split_indices,
        train_size=TRAIN_SPLIT,
        test_size=TEST_SPLIT,
        stratify=labels,
        random_state=RANDOM_SEED,
    )

    x_train, x_test = features[train_idx], features[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    print(f"Training samples: {len(train_idx)}")
    print(f"Test samples: {len(test_idx)}")

    classifier = train_classifier(x_train, y_train)
    metrics = evaluate_classifier(classifier, x_train, y_train, x_test, y_test, class_specs)
    metrics["pca_components_selected"] = int(classifier.named_steps["pca"].n_components_)
    metrics["pca_variance_explained"] = float(np.sum(classifier.named_steps["pca"].explained_variance_ratio_))

    metrics["dataset"] = {
        "profile": args.dataset_profile,
        "display_name": display_name,
        "data_root": str(data_root),
        "image_size": list(IMAGE_SIZE),
        "num_images": len(images),
        "class_counts": {spec.display_name: int(label_counts[spec.label]) for spec in class_specs},
    }
    metrics["split"] = {
        "train_indices": train_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "train_class_counts": {spec.display_name: int((y_train == spec.label).sum()) for spec in class_specs},
        "test_class_counts": {spec.display_name: int((y_test == spec.label).sum()) for spec in class_specs},
    }
    metrics["feature_vector_length"] = int(features.shape[1])
    metrics["files"] = {
        "train": [str(paths[index]) for index in train_idx],
        "test": [str(paths[index]) for index in test_idx],
    }

    metrics_path = RESULTS_DIR / f"{result_prefix}_metrics.json"
    confusion_path = RESULTS_DIR / f"{result_prefix}_confusion_matrix.png"

    save_metrics(metrics, metrics_path)
    plot_confusion_matrix(np.asarray(metrics["confusion_matrix"]), class_specs, confusion_path)
    print_phase0_results(metrics, metrics_path, confusion_path)


if __name__ == "__main__":
    main()
