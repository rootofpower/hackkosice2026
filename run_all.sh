#!/bin/bash
set -e

# Hard pairs get B2 backbone; easy pairs get B0
HARD_PAIRS=("chory" "diabetes" "skleroza")
EASY_PAIRS=("pgov" "suche_oko")

# Use 'python' — adjust to your env (e.g. 'python3' or '.venv/bin/python')
PY="${PYTHON:-python}"

echo "============================================"
echo "  Using Python: $PY"
echo "============================================"

echo ""
echo "============================================"
echo "  Phase 0: Data integrity check"
echo "============================================"
$PY check_duplicates.py

echo ""
echo "============================================"
echo "  Phase 1: Train hard pairs with B2"
echo "============================================"
for PAIR in "${HARD_PAIRS[@]}"; do
    echo "========================================"
    echo "Training pair: $PAIR (backbone=b2, seed=42)"
    echo "========================================"
    $PY efficientnet_v2.py --pair $PAIR --backbone b2 --seed 42

    echo "Training pair: $PAIR (backbone=b2, seed=123)"
    $PY efficientnet_v2.py --pair $PAIR --backbone b2 --seed 123

    echo "Ensemble evaluation: $PAIR"
    $PY fn_ensemble_v2.py --pair $PAIR --backbone b2

    echo "Done: $PAIR"
done

echo ""
echo "============================================"
echo "  Phase 2: Train easy pairs with B0"
echo "============================================"
for PAIR in "${EASY_PAIRS[@]}"; do
    echo "========================================"
    echo "Training pair: $PAIR (backbone=b0, seed=42)"
    echo "========================================"
    $PY efficientnet_v2.py --pair $PAIR --backbone b0 --seed 42

    echo "Ensemble evaluation: $PAIR"
    $PY fn_ensemble_v2.py --pair $PAIR --backbone b0

    echo "Done: $PAIR"
done

echo ""
echo "============================================"
echo "  All pairs complete."
echo "============================================"
