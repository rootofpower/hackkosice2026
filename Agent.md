You are an expert ML engineer specializing in medical image analysis. Help me build a complete image classification pipeline for microscopy images of human tears.

## Project context
- Task: classify tear microscopy images into 5 classes
- Classes and approximate sample counts:
  - ZdraviLudia (healthy): ~100 images
  - SklerózaMultiplex: ~100 images
  - Diabetes: ~30 images
  - PGOV_Glaukom: ~30 images
  - SucheOko: ~30 images
- Dataset is small and class-imbalanced
- Images are already preprocessed:
  - Grayscale (single channel)
  - Fixed resolution: 523×490 px
  - Uniform format (no further preprocessing needed)
  - Do NOT add any resizing, color conversion, or normalization steps
    that assume a different input format

## Phase 0 — Binary prototype (run this FIRST, use "alpha_model" directory)
Before the full 5-class pipeline, implement and evaluate a simplified
binary classifier using only:
  - Class 0: chore_suche — 12 images
  - Class 1: zdrave — 12 images
  Total: 24 images

This prototype validates that the pipeline works end-to-end before
scaling to the full dataset. Apply the same feature extraction and
classifier logic as in Approach 1 below, but with:
  - Stratified 80/20 train/test split (no cross-validation at this stage)
  - Binary metrics: accuracy, F1, confusion matrix, ROC-AUC
  - Clear printed output: "Phase 0 results: ..."

## Requirements
Build TWO parallel approaches so I can compare them:

### Approach 1: Feature Extraction + Random Forest
1. Feature extraction per image (input: 523×490 grayscale numpy array):
   - LBP (Local Binary Pattern) — texture descriptor
     params: P=16, R=2, method='uniform'
   - GLCM (Gray-Level Co-occurrence Matrix) — structural regularity
     features: contrast, energy, homogeneity, correlation
     params: distances=[1,2], angles=[0, pi/4, pi/2, 3pi/4]
   - HOG (Histogram of Oriented Gradients) — shape features
     params: orientations=9, pixels_per_cell=(16,16), block_size=(2,2)
   - Basic shape features via contour analysis:
     area, perimeter, circularity, solidity of largest contour

2. Combine all features into a single feature vector per image

3. Classifier:
   - RandomForestClassifier(n_estimators=300, class_weight='balanced')
   - Hyperparameter tuning with RandomizedSearchCV
   - Stratified K-Fold cross-validation (k=5)

### Approach 2: Transfer Learning (EfficientNet-B0)
1. Augmentation pipeline (for minority classes only — 30-image classes):
   - RandomHorizontalFlip, RandomVerticalFlip
   - RandomRotation(180) — microscopy has no fixed orientation
   - RandomAffine(scale=(0.8, 1.2))
   - ColorJitter(brightness=0.2, contrast=0.3)
   - Do NOT resize — keep 523×490
   - Normalize with ImageNet mean/std (replicate grayscale to 3 channels)
   - Target: expand 30-image classes to ~150+ via augmentation

2. Model setup:
   - EfficientNet-B0 pretrained on ImageNet (torchvision)
   - Adapt first conv layer to accept 1-channel input OR replicate
     grayscale to 3 channels before passing to the model
   - Freeze all layers except classifier head
   - Replace classifier: nn.Linear(1280, 5)
   - Loss: CrossEntropyLoss(weight=class_weights)
   - Optimizer: AdamW(lr=1e-3, weight_decay=1e-4)
   - Scheduler: CosineAnnealingLR

3. Training:
   - Stratified train/test split: 80% train, 20% test
   - Early stopping (patience=10)
   - Save best checkpoint by val F1-macro

## Evaluation (both approaches)
Report these metrics per class AND macro-averaged:
- F1-score (primary metric — NOT accuracy due to imbalance)
- Precision, Recall
- Confusion matrix (with class labels in Slovak)
- ROC-AUC (one-vs-rest)

## Code structure
Organize as a single well-commented Python script or Jupyter notebook:
1. Config (paths, hyperparams as constants at top)
   - IMAGE_SIZE = (523, 490)
   - TRAIN_SPLIT = 0.80
   - TEST_SPLIT = 0.20
2. Data loading (no preprocessing — images are ready)
3. Phase 0: binary prototype (SucheOko vs ZdraviLudia, 12+12)
4. Approach 1: feature extraction pipeline
5. Approach 1: training + evaluation
6. Approach 2: augmentation + dataloader
7. Approach 2: training loop + evaluation
8. Side-by-side comparison of both approaches

## Constraints
- Use scikit-learn, OpenCV, PyTorch, torchvision
- Include requirements.txt
- Handle missing/corrupt images gracefully
- Add progress bars (tqdm)
- Save all results to a results/ folder (metrics JSON + confusion matrix PNG)
- Code must run on CPU (no GPU assumption)

make no mistakes.