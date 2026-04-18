#!/usr/bin/env python3
"""Binary classification pipeline using datasets from the test/ directory.

Usage examples:
    # Default: zdravi_ludi vs diabetes_spolu_zo_suchym_okom
    python test_pipeline.py

    # Custom two-class selection:
    python test_pipeline.py --class0 suche_oko --class1 zdravi_ludi

    # List available classes in the test/ directory:
    python test_pipeline.py --list-classes
"""

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
from skimage.filters import gabor, threshold_otsu
from skimage.measure import euler_number, label, regionprops
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Test images are 490×490 (post-processing cropped them to square)
IMAGE_SIZE = (490, 490)  # (width, height) — stored as (H, W) by OpenCV
TRAIN_SPLIT = 0.80
TEST_SPLIT = 0.20
RANDOM_SEED = 42

TEST_DATA_ROOT = Path(__file__).resolve().parent / "test"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

SUPPORTED_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# Feature extraction hyper-parameters (same as train_pipeline.py)
LBP_POINTS = 16
LBP_RADIUS = 2
LBP_METHOD = "uniform"

GLCM_DISTANCES = [1, 2]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_PROPERTIES = ("contrast", "energy", "homogeneity", "correlation")

HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

# Gabor filter parameters (crystalline orientation patterns)
GABOR_FREQUENCIES = [0.1, 0.2, 0.3, 0.4, 0.5]
GABOR_ORIENTATIONS = [i * np.pi / 8 for i in range(8)]
GABOR_DOWNSCALE = 128  # downscale image for Gabor (stats are scale-invariant)

# Fractal dimension box sizes
FRACTAL_BOX_SIZES = 2 ** np.arange(2, 8)  # 4, 8, 16, 32, 64, 128


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ClassSpec:
    label: int
    folder_name: str
    display_name: str


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def discover_available_classes(data_root: Path) -> list[str]:
    """Return sorted list of subdirectory names under *data_root*."""
    if not data_root.is_dir():
        raise FileNotFoundError(f"Test data root does not exist: {data_root}")
    return sorted(
        entry.name
        for entry in data_root.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    )


def _find_image_dir(class_dir: Path) -> Path:
    """Resolve the actual directory containing images.

    The test/ layout nests images one level deeper, e.g.:
        test/zdravi_ludi/super_proccesed/*.bmp
        test/diabetes_spolu_zo_suchym_okom/final_dataset/*.bmp

    This helper returns the first sub-directory that contains image files.
    If the class_dir itself contains images, it returns class_dir.
    """
    # Check if images are directly in class_dir
    direct_images = [
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if direct_images:
        return class_dir

    # Look one level deeper
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


def build_class_specs(class0_name: str, class1_name: str) -> tuple[ClassSpec, ...]:
    """Build a pair of ClassSpec objects from folder names."""
    return (
        ClassSpec(label=0, folder_name=class0_name, display_name=class0_name),
        ClassSpec(label=1, folder_name=class1_name, display_name=class1_name),
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def expected_image_shape() -> tuple[int, int]:
    """OpenCV grayscale shape is (height, width)."""
    return IMAGE_SIZE[1], IMAGE_SIZE[0]


def collect_image_paths(
    data_root: Path, class_specs: tuple[ClassSpec, ...]
) -> list[tuple[Path, ClassSpec]]:
    """Collect all image paths for the given class specs, handling nested dirs."""
    samples: list[tuple[Path, ClassSpec]] = []

    for spec in class_specs:
        class_dir = data_root / spec.folder_name
        if not class_dir.exists():
            warnings.warn(f"Missing class directory: {class_dir}")
            continue

        try:
            image_dir = _find_image_dir(class_dir)
        except FileNotFoundError as exc:
            warnings.warn(str(exc))
            continue

        image_paths = sorted(
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not image_paths:
            warnings.warn(f"No images for class '{spec.display_name}' in {image_dir}")

        samples.extend((path, spec) for path in image_paths)

    return samples


def load_grayscale_image(image_path: Path) -> np.ndarray | None:
    """Load a single grayscale image, validating shape."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        warnings.warn(f"Skipping unreadable image: {image_path}")
        return None

    if image.ndim != 2:
        warnings.warn(f"Skipping non-grayscale image: {image_path}")
        return None

    expected = expected_image_shape()
    if image.shape != expected:
        warnings.warn(
            f"Skipping image with unexpected shape {image.shape}: {image_path}. "
            f"Expected {expected}."
        )
        return None

    return image


def load_dataset(
    data_root: Path, class_specs: tuple[ClassSpec, ...]
) -> tuple[list[np.ndarray], np.ndarray, list[Path]]:
    """Load all images for the given classes."""
    samples = collect_image_paths(data_root, class_specs)
    if not samples:
        raise RuntimeError(
            f"No readable image candidates found under {data_root.resolve()}."
        )

    images: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[Path] = []

    for image_path, spec in tqdm(samples, desc="Loading images", unit="image"):
        image = load_grayscale_image(image_path)
        if image is None:
            continue
        images.append(image)
        labels.append(spec.label)
        paths.append(image_path)

    label_counts = Counter(labels)
    missing = [
        spec.display_name
        for spec in class_specs
        if label_counts.get(spec.label, 0) == 0
    ]
    if missing:
        raise RuntimeError(
            f"Cannot run: these classes have no valid images: {', '.join(missing)}"
        )

    return images, np.asarray(labels, dtype=np.int64), paths


# ---------------------------------------------------------------------------
# Feature extraction (identical to train_pipeline.py)
# ---------------------------------------------------------------------------

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
    _, binary_mask = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    contour_result = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = contour_result[0] if len(contour_result) == 2 else contour_result[1]

    if not contours:
        return np.zeros(4, dtype=np.float32)

    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    perimeter = float(cv2.arcLength(largest, closed=True))
    circularity = (4.0 * math.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

    hull = cv2.convexHull(largest)
    hull_area = float(cv2.contourArea(hull))
    solidity = (area / hull_area) if hull_area > 0 else 0.0

    return np.asarray([area, perimeter, circularity, solidity], dtype=np.float32)


def extract_gabor_features(image: np.ndarray) -> np.ndarray:
    """Gabor filter responses — capture crystalline orientation patterns.

    5 frequencies × 8 orientations = 40 filter responses.
    For each response: mean, std, max of the magnitude → 120 features.
    Downscaled to 128×128 for speed (orientation statistics are scale-invariant).
    """
    # Downscale and normalise to [0, 1] float64 to avoid overflow
    small = cv2.resize(image, (GABOR_DOWNSCALE, GABOR_DOWNSCALE), interpolation=cv2.INTER_AREA)
    small = small.astype(np.float64) / 255.0

    features: list[float] = []
    for freq in GABOR_FREQUENCIES:
        for theta in GABOR_ORIENTATIONS:
            real, imag = gabor(small, frequency=freq, theta=theta)
            magnitude = np.sqrt(real ** 2 + imag ** 2)
            features.extend([
                float(magnitude.mean()),
                float(magnitude.std()),
                float(magnitude.max()),
            ])
    return np.asarray(features, dtype=np.float32)  # 120 features


def extract_fractal_dimension(image: np.ndarray) -> np.ndarray:
    """Box-counting fractal dimension — measures structural complexity.

    Higher values indicate more complex crystalline branching patterns.
    Returns 1 feature: the fractal dimension estimate.
    Vectorised box counting for performance.
    """
    thresh = threshold_otsu(image)
    binary = image > thresh
    h, w = binary.shape

    counts: list[int] = []
    for size in FRACTAL_BOX_SIZES:
        # Trim image to exact multiple of box size, then reshape into boxes
        nh = (h // size) * size
        nw = (w // size) * size
        trimmed = binary[:nh, :nw]
        # Reshape into (num_boxes_h, size, num_boxes_w, size) then check any
        boxes = trimmed.reshape(nh // size, size, nw // size, size)
        counts.append(int(boxes.any(axis=(1, 3)).sum()))

    # Fit line in log-log space → slope ≈ fractal dimension
    log_sizes = np.log(FRACTAL_BOX_SIZES.astype(np.float64))
    log_counts = np.log(np.asarray(counts, dtype=np.float64) + 1e-12)
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    return np.asarray([abs(coeffs[0])], dtype=np.float32)  # 1 feature


def extract_topology_features(image: np.ndarray) -> np.ndarray:
    """Topological features — crystal fragment structure analysis.

    Analyses connected components on a binarised image to capture:
    - number of crystal fragments
    - mean / std / max fragment area
    - Euler number (topological complexity)
    Returns 5 features.
    """
    thresh = threshold_otsu(image)
    binary = image > thresh

    labeled = label(binary)
    props = regionprops(labeled)

    num_components = len(props)
    areas = [float(p.area) for p in props] if props else [0.0]
    euler = float(euler_number(binary))

    return np.asarray(
        [
            num_components,                                     # total crystal fragments
            float(np.mean(areas)),                              # avg fragment size
            float(np.std(areas)) if len(areas) > 1 else 0.0,   # size variability
            float(np.max(areas)),                               # largest fragment
            euler,                                              # topological complexity
        ],
        dtype=np.float32,
    )  # 5 features


def extract_feature_vector(image: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            # Original features
            extract_lbp_features(image),
            extract_glcm_features(image),
            extract_hog_features(image),
            extract_shape_features(image),
            # New domain-specific features
            extract_gabor_features(image),       # +120
            extract_fractal_dimension(image),    # +1
            extract_topology_features(image),    # +5
        ]
    ).astype(np.float32)


def extract_feature_matrix(images: list[np.ndarray]) -> np.ndarray:
    rows = [
        extract_feature_vector(img)
        for img in tqdm(images, desc="Extracting features", unit="image")
    ]
    lengths = {r.shape[0] for r in rows}
    if len(lengths) != 1:
        raise RuntimeError(
            f"Inconsistent feature vector lengths: {sorted(lengths)}"
        )
    return np.vstack(rows)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_classifier(x_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train
    )
    class_weight_dict = dict(zip(classes.tolist(), weights.tolist()))
    pca_components = min(100, x_train.shape[0], x_train.shape[1])

    clf = Pipeline(
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
    clf.fit(x_train, y_train)
    return clf


# ---------------------------------------------------------------------------
# Evaluation — dual threshold strategies
# ---------------------------------------------------------------------------

TARGET_RECALL = 0.80  # Strategy 1: minimum recall for disease class


def _evaluate_at_threshold(
    y_test: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    ordered_labels: list[int],
    display_names: list[str],
) -> dict[str, Any]:
    """Produce metrics for a single threshold."""
    y_pred = np.where(y_proba >= threshold, 0, 1)

    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=ordered_labels, zero_division=0,
    )
    report = classification_report(
        y_test, y_pred, labels=ordered_labels,
        target_names=display_names, zero_division=0,
    )

    result: dict[str, Any] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=ordered_labels).tolist(),
        "classification_report": report,
        "per_class": {},
        "macro_avg": {
            "precision": float(np.mean(precision)),
            "recall": float(np.mean(recall)),
            "f1": float(np.mean(per_class_f1)),
        },
    }
    for idx, name in enumerate(display_names):
        result["per_class"][name] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(per_class_f1[idx]),
            "support": int(support[idx]),
        }
    return result


def evaluate_classifier(
    classifier: Pipeline,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_specs: tuple[ClassSpec, ...],
) -> dict[str, Any]:
    ordered_labels = [s.label for s in class_specs]
    display_names = [s.display_name for s in class_specs]

    # --- Compute ROC on the TRAINING set to find thresholds ---------------
    train_prob_class0 = classifier.predict_proba(x_train)[:, 0]
    fpr, tpr, thresholds = roc_curve(y_train, train_prob_class0, pos_label=0)

    # Strategy 1 — minimum threshold where recall (TPR) >= TARGET_RECALL
    valid_mask = tpr >= TARGET_RECALL
    if valid_mask.any():
        # Among all thresholds meeting the recall target, pick the highest
        # (most conservative that still meets recall) to avoid over-lowering.
        threshold_recall = float(thresholds[valid_mask][ 0])  # highest threshold
    else:
        threshold_recall = 0.5
        warnings.warn(
            f"No threshold achieves recall >= {TARGET_RECALL:.0%} on train set; "
            "defaulting to 0.5."
        )

    # Strategy 2 — maximize F1 for disease class (class 0) only
    # F1 = 2·TP / (2·TP + FP + FN) = 2·TPR / (2·TPR + FPR + (1-TPR))
    #    = 2·TPR / (1 + TPR + FPR)
    f1_disease = 2 * tpr / (1 + tpr + fpr + 1e-8)
    threshold_f1 = float(thresholds[int(np.argmax(f1_disease))])

    # --- Also keep Youden's J for reference (not primary) -----------------
    j_scores = tpr - fpr
    threshold_youden = float(thresholds[int(np.argmax(j_scores))])

    # --- Test-set probabilities -------------------------------------------
    y_proba = classifier.predict_proba(x_test)[:, 0]
    class0_probs = y_proba[y_test == 0]

    # --- ROC-AUC (threshold-independent) ----------------------------------
    roc_auc_value = None
    try:
        positive_mask = (y_test == ordered_labels[0]).astype(np.int64)
        roc_auc_value = float(roc_auc_score(positive_mask, y_proba))
    except ValueError as err:
        warnings.warn(f"ROC-AUC could not be computed: {err}")

    # --- Evaluate at BOTH thresholds + Youden reference -------------------
    eval_recall = _evaluate_at_threshold(
        y_test, y_proba, threshold_recall, ordered_labels, display_names,
    )
    eval_f1 = _evaluate_at_threshold(
        y_test, y_proba, threshold_f1, ordered_labels, display_names,
    )
    eval_youden = _evaluate_at_threshold(
        y_test, y_proba, threshold_youden, ordered_labels, display_names,
    )

    metrics: dict[str, Any] = {
        "roc_auc": roc_auc_value,
        "class0_predicted_probs": class0_probs.round(3).tolist(),
        "class_names": display_names,
        "thresholds": {
            "recall_ge_80": eval_recall,
            "max_f1_disease": eval_f1,
            "youden_j_reference": eval_youden,
        },
    }

    return metrics


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def plot_confusion_matrices(
    metrics: dict[str, Any],
    class_specs: tuple[ClassSpec, ...],
    output_path: Path,
) -> None:
    """Plot confusion matrices for all threshold strategies side-by-side."""
    names = [s.display_name for s in class_specs]
    strategies = metrics["thresholds"]
    titles = {
        "recall_ge_80": f"Recall ≥ {TARGET_RECALL:.0%}",
        "max_f1_disease": "Max F1 (disease)",
        "youden_j_reference": "Youden's J (ref)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (key, strat) in zip(axes, strategies.items()):
        cm = np.asarray(strat["confusion_matrix"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=names, yticklabels=names, ax=ax,
        )
        ax.set_xlabel("Predikovane")
        ax.set_ylabel("Skutocne")
        ax.set_title(f"{titles[key]}\nthreshold = {strat['threshold']:.3f}")
    fig.suptitle("Threshold Strategy Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)


def _print_strategy_block(label: str, strat: dict[str, Any]) -> None:
    """Print metrics for one threshold strategy."""
    print(f"\n  ── {label} ──")
    print(f"     Threshold : {strat['threshold']:.3f}")
    print(f"     Accuracy  : {strat['accuracy']:.4f}")
    print(f"     F1-macro  : {strat['f1_macro']:.4f}")
    print(f"     Confusion matrix:")
    for row in strat["confusion_matrix"]:
        print(f"       {row}")
    print(f"     Classification report:")
    for line in strat["classification_report"].splitlines():
        print(f"       {line}")


def print_results(
    metrics: dict[str, Any], metrics_path: Path, confusion_path: Path
) -> None:
    print("\n" + "=" * 70)
    print("TEST PIPELINE RESULTS — THRESHOLD COMPARISON")
    print("=" * 70)
    print(f"  Classes : {metrics['class_names']}")
    if metrics["roc_auc"] is None:
        print("  ROC-AUC : unavailable")
    else:
        print(f"  ROC-AUC : {metrics['roc_auc']:.4f}  (threshold-independent)")
    print(f"  PCA     : {metrics['pca_components_selected']} components, "
          f"{metrics['pca_variance_explained']:.1%} variance explained")

    strats = metrics["thresholds"]
    recall_t = strats['recall_ge_80']['threshold']
    f1_t = strats['max_f1_disease']['threshold']
    print(f"\n  Threshold (recall≥{TARGET_RECALL:.0%}): {recall_t:.3f}")
    print(f"  Threshold (max F1 disease) : {f1_t:.3f}")

    _print_strategy_block(
        f"Strategy 1 — Recall ≥ {TARGET_RECALL:.0%} (minimize false negatives)",
        strats["recall_ge_80"],
    )
    _print_strategy_block(
        "Strategy 2 — Max F1 for disease class",
        strats["max_f1_disease"],
    )
    _print_strategy_block(
        "Reference — Youden's J (old behaviour)",
        strats["youden_j_reference"],
    )

    print(f"\n  Metrics JSON → {metrics_path}")
    print(f"  Confusion PNG → {confusion_path}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Binary classification on test/ datasets. "
        "Select two classes from test/ subdirectories."
    )
    parser.add_argument(
        "--class0",
        default="diabetes_spolu_zo_suchym_okom",
        help="Folder name for class 0 (positive / disease). "
        "Default: diabetes_spolu_zo_suchym_okom",
    )
    parser.add_argument(
        "--class1",
        default="zdravi_ludi",
        help="Folder name for class 1 (negative / healthy). "
        "Default: zdravi_ludi",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override the test data root directory. Default: <project>/test",
    )
    parser.add_argument(
        "--list-classes",
        action="store_true",
        help="Print available class folders and exit.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    data_root = args.data_root or TEST_DATA_ROOT

    if args.list_classes:
        classes = discover_available_classes(data_root)
        print("Available class folders in", data_root)
        for name in classes:
            print(f"  • {name}")
        return

    class_specs = build_class_specs(args.class0, args.class1)
    result_tag = f"test_{args.class0}_vs_{args.class1}"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Data root  : {data_root.resolve()}")
    print(f"Class 0    : {args.class0}")
    print(f"Class 1    : {args.class1}")
    print(f"Image size : {IMAGE_SIZE}")
    print()

    # --- Load ---
    images, labels, paths = load_dataset(data_root, class_specs)

    label_counts = Counter(labels.tolist())
    print("Images per class:")
    for spec in class_specs:
        print(f"  {spec.display_name}: {label_counts[spec.label]}")

    # --- Features ---
    features = extract_feature_matrix(images)
    print(f"Feature matrix shape: {features.shape}")

    # --- Split ---
    indices = np.arange(len(paths))
    train_idx, test_idx = train_test_split(
        indices,
        train_size=TRAIN_SPLIT,
        test_size=TEST_SPLIT,
        stratify=labels,
        random_state=RANDOM_SEED,
    )

    x_train, x_test = features[train_idx], features[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    print(f"Training samples : {len(train_idx)}")
    print(f"Test samples     : {len(test_idx)}")

    # --- Train ---
    classifier = train_classifier(x_train, y_train)

    # --- Evaluate ---
    metrics = evaluate_classifier(
        classifier, x_train, y_train, x_test, y_test, class_specs
    )
    metrics["pca_components_selected"] = int(
        classifier.named_steps["pca"].n_components_
    )
    metrics["pca_variance_explained"] = float(
        np.sum(classifier.named_steps["pca"].explained_variance_ratio_)
    )

    metrics["dataset"] = {
        "class0": args.class0,
        "class1": args.class1,
        "data_root": str(data_root),
        "image_size": list(IMAGE_SIZE),
        "num_images": len(images),
        "class_counts": {
            spec.display_name: int(label_counts[spec.label])
            for spec in class_specs
        },
    }
    metrics["split"] = {
        "train_indices": train_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "train_class_counts": {
            spec.display_name: int((y_train == spec.label).sum())
            for spec in class_specs
        },
        "test_class_counts": {
            spec.display_name: int((y_test == spec.label).sum())
            for spec in class_specs
        },
    }
    metrics["feature_vector_length"] = int(features.shape[1])
    metrics["files"] = {
        "train": [str(paths[i]) for i in train_idx],
        "test": [str(paths[i]) for i in test_idx],
    }

    # --- Save ---
    metrics_path = RESULTS_DIR / f"{result_tag}_metrics.json"
    confusion_path = RESULTS_DIR / f"{result_tag}_confusion_matrix.png"

    save_metrics(metrics, metrics_path)
    plot_confusion_matrices(metrics, class_specs, confusion_path)
    print_results(metrics, metrics_path, confusion_path)

    # --- Before / After comparison (baseline from previous run) -----------
    BASELINE = {
        "ROC-AUC": 0.8544,
        "F1-macro (max-F1)": 0.6745,
        "Disease recall (max-F1)": 0.45,
        "Feature columns": 30330,
    }
    max_f1_strat = metrics["thresholds"]["max_f1_disease"]
    disease_recall = max_f1_strat["per_class"][class_specs[0].display_name]["recall"]
    AFTER = {
        "ROC-AUC": metrics["roc_auc"] or 0.0,
        "F1-macro (max-F1)": max_f1_strat["f1_macro"],
        "Disease recall (max-F1)": disease_recall,
        "Feature columns": features.shape[1],
    }
    print("\n" + "=" * 55)
    print("BEFORE vs AFTER — new feature groups")
    print("=" * 55)
    print(f"{'Metric':<25} {'Before':>12} {'After':>12}")
    print("-" * 55)
    for key in BASELINE:
        bval = BASELINE[key]
        aval = AFTER[key]
        if isinstance(bval, int):
            print(f"{key:<25} {bval:>12d} {aval:>12d}")
        else:
            print(f"{key:<25} {bval:>12.4f} {aval:>12.4f}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
