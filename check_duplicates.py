#!/usr/bin/env python3
"""Check for data leakage: duplicate/near-duplicate images between train and test sets."""

import hashlib
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

from config import PAIRS

SUPPORTED_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def file_hash(path: str) -> str:
    """MD5 hash of raw file bytes."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def pixel_hash(path: str, size=(64, 64)) -> str:
    """Hash of resized pixel content — catches re-encoded duplicates."""
    try:
        img = Image.open(path).convert("L").resize(size, Image.BILINEAR)
        arr = np.array(img, dtype=np.uint8)
        return hashlib.md5(arr.tobytes()).hexdigest()
    except Exception:
        return "ERROR"


def collect_images(directory: str):
    """Collect all image paths from a directory."""
    paths = []
    for p in sorted(Path(directory).glob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            paths.append(str(p))
    return paths


def check_pair(pair_key: str):
    pair = PAIRS[pair_key]
    
    train_disease = collect_images(pair["disease_train"])
    train_healthy = collect_images(pair["healthy_train"])
    test_disease = collect_images(pair["disease_test"])
    test_healthy = collect_images(pair["healthy_test"])
    
    train_all = train_disease + train_healthy
    test_all = test_disease + test_healthy
    
    print(f"\n{'='*70}")
    print(f"  {pair_key.upper()} — {pair['label']}")
    print(f"{'='*70}")
    print(f"  Train: {len(train_disease)} disease + {len(train_healthy)} healthy = {len(train_all)}")
    print(f"  Test:  {len(test_disease)} disease + {len(test_healthy)} healthy = {len(test_all)}")
    
    # 1. Exact file hash duplicates
    print(f"\n  --- Exact file hash check ---")
    train_hashes = {}
    for p in train_all:
        h = file_hash(p)
        train_hashes[h] = p
    
    exact_dupes = 0
    for p in test_all:
        h = file_hash(p)
        if h in train_hashes:
            exact_dupes += 1
            print(f"    EXACT DUPLICATE: test={Path(p).name} <-> train={Path(train_hashes[h]).name}")
    
    if exact_dupes == 0:
        print(f"    No exact duplicates found.")
    else:
        print(f"    ⚠ {exact_dupes} exact duplicates found!")
    
    # 2. Pixel-level hash duplicates (catches re-encoded images)
    print(f"\n  --- Pixel content hash check (64x64 resize) ---")
    train_pixel_hashes = {}
    for p in train_all:
        h = pixel_hash(p)
        train_pixel_hashes[h] = p
    
    pixel_dupes = 0
    for p in test_all:
        h = pixel_hash(p)
        if h in train_pixel_hashes:
            pixel_dupes += 1
            if pixel_dupes <= 10:  # only print first 10
                print(f"    PIXEL DUPLICATE: test={Path(p).name} <-> train={Path(train_pixel_hashes[h]).name}")
    
    if pixel_dupes == 0:
        print(f"    No pixel duplicates found.")
    else:
        print(f"    ⚠ {pixel_dupes} pixel-level duplicates found!")
    
    # 3. Check within-set duplicates
    print(f"\n  --- Within-set duplicate check ---")
    train_hash_counts = defaultdict(list)
    for p in train_all:
        h = pixel_hash(p)
        train_hash_counts[h].append(p)
    
    within_train = sum(1 for v in train_hash_counts.values() if len(v) > 1)
    if within_train > 0:
        print(f"    ⚠ {within_train} duplicate groups within TRAIN set")
        for h, paths in train_hash_counts.items():
            if len(paths) > 1:
                names = [Path(p).name for p in paths[:3]]
                print(f"      {names} ({len(paths)} copies)")
    else:
        print(f"    No within-train duplicates.")
    
    test_hash_counts = defaultdict(list)
    for p in test_all:
        h = pixel_hash(p)
        test_hash_counts[h].append(p)
    
    within_test = sum(1 for v in test_hash_counts.values() if len(v) > 1)
    if within_test > 0:
        print(f"    ⚠ {within_test} duplicate groups within TEST set")
    else:
        print(f"    No within-test duplicates.")
    
    # 4. Filename overlap check
    print(f"\n  --- Filename overlap check ---")
    train_names = {Path(p).name for p in train_all}
    test_names = {Path(p).name for p in test_all}
    overlap = train_names & test_names
    if overlap:
        print(f"    ⚠ {len(overlap)} filenames appear in both train and test:")
        for name in sorted(list(overlap))[:10]:
            print(f"      {name}")
    else:
        print(f"    No filename overlap.")
    
    return exact_dupes, pixel_dupes


def main():
    pairs_to_check = sys.argv[1:] if len(sys.argv) > 1 else list(PAIRS.keys())
    
    print("=" * 70)
    print("  DATA INTEGRITY CHECK — Train/Test Leakage Detection")
    print("=" * 70)
    
    results = {}
    for pair_key in pairs_to_check:
        if pair_key not in PAIRS:
            print(f"Unknown pair: {pair_key}")
            continue
        exact, pixel = check_pair(pair_key)
        results[pair_key] = (exact, pixel)
    
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    for pair_key, (exact, pixel) in results.items():
        status = "✓ CLEAN" if exact == 0 and pixel == 0 else "⚠ LEAKAGE"
        print(f"  {pair_key:15s} | {status} | exact={exact}, pixel={pixel}")


if __name__ == "__main__":
    main()
