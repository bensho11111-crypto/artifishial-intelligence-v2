#!/bin/bash
# Evaluate all 4 trained models with captain agents

cd "$(dirname "$0")"

echo "=== Evaluating all 4 experiment models ==="
echo ""

# Function to evaluate a model
eval_model() {
    local name=$1
    local checkpoint=$2
    local out_file=$3

    echo "Evaluating $name..."
    python tools/eval_model.py \
        --model "$checkpoint" \
        --episodes 10 \
        --duration 300 \
        --out "$out_file"

    if [ -f "$out_file" ]; then
        echo "✓ $name evaluation complete: $out_file"
    else
        echo "✗ $name evaluation failed"
    fi
    echo ""
}

# Evaluate each model
eval_model "exp/larger-dataset (baseline)" \
    "checkpoints_large/best.pt" \
    "eval_larger_dataset.json" &

eval_model "exp/approach-detection (proximity labels)" \
    "checkpoints_approach/best.pt" \
    "eval_approach_detection.json" &

eval_model "exp/loss-tuning (aggressive loss)" \
    "checkpoints_loss_tuning/best.pt" \
    "eval_loss_tuning.json" &

eval_model "exp/class-resampling (weighted sampler)" \
    "checkpoints_resampling/best.pt" \
    "eval_class_resampling.json" &

wait

echo "=== All evaluations complete ==="
