#!/bin/bash
set -e

PAIRS=("diabetes" "pgov" "skleroza")

for PAIR in "${PAIRS[@]}"; do
    echo "========================================"
    echo "Training pair: $PAIR"
    echo "========================================"
    python efficientnet_train.py    --pair $PAIR
    python efficientnet_finetune.py --pair $PAIR
    python efficientnet_focal.py    --pair $PAIR
    python fn_ensemble.py           --pair $PAIR
    echo "Done: $PAIR"
done

echo "All pairs complete."
