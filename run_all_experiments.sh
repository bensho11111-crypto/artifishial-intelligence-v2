#!/bin/bash
# Run all 4 training experiments for label sparsity solutions

set -e

echo "========================================="
echo "4-EXPERIMENT CAMPAIGN: LABEL SPARSITY"
echo "========================================="
echo ""

# Experiment 1: exp/larger-dataset (baseline with 300 sessions)
echo "[1/4] exp/larger-dataset (baseline, 300 sessions)"
git checkout exp/larger-dataset
python tools/train_model.py \
    --data data/training_300/ \
    --out checkpoints_300/ \
    --epochs 15 \
    --batch 32
echo "✓ Checkpoint saved to checkpoints_300/best.pt"
echo ""

# Experiment 2: exp/approach-detection (proximity labels)
echo "[2/4] exp/approach-detection (proximity labels, 300 sessions)"
git checkout exp/approach-detection
python tools/train_model.py \
    --data data/training_300/ \
    --out checkpoints_approach_detection/ \
    --epochs 15 \
    --batch 32
echo "✓ Checkpoint saved to checkpoints_approach_detection/best.pt"
echo ""

# Experiment 3: exp/loss-tuning (aggressive focal loss)
echo "[3/4] exp/loss-tuning (gamma_neg=8.0, 300 sessions)"
git checkout exp/loss-tuning
python tools/train_model.py \
    --data data/training_300/ \
    --out checkpoints_loss_tuning/ \
    --epochs 15 \
    --batch 32
echo "✓ Checkpoint saved to checkpoints_loss_tuning/best.pt"
echo ""

# Experiment 4: exp/class-resampling (weighted sampler)
echo "[4/4] exp/class-resampling (weighted sampler, 300 sessions)"
git checkout exp/class-resampling
python tools/train_model.py \
    --data data/training_300/ \
    --out checkpoints_class_resampling/ \
    --epochs 15 \
    --batch 32
echo "✓ Checkpoint saved to checkpoints_class_resampling/best.pt"
echo ""

echo "========================================="
echo "All 4 experiments complete!"
echo "Next: Run evaluation suite"
echo "  bash eval_all_experiments.sh"
echo "  python compare_experiments.py"
echo "========================================="
