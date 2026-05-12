#!/bin/bash
# Evaluate the newly trained model and compare with old results

cd "C:\Users\Ben\Desktop\artifishial_intelligence_v2"

echo "======================================================================"
echo "EVALUATING NEWLY TRAINED MODEL"
echo "======================================================================"

# Wait for training to complete
until [ -f "checkpoints_real_synthetic/best.pt" ] && grep -q "TRAINING COMPLETE" train_fixed.log; do
    echo "Waiting for training to complete..."
    sleep 10
done

echo "Training complete. Evaluating model with seeking captain..."
python tools/eval_proximity.py \
    --model checkpoints_real_synthetic/best.pt \
    --duration 120 \
    --seeds 0,1,2 \
    --out eval_seeking_captain_final.json

echo ""
echo "======================================================================"
echo "RESULTS"
echo "======================================================================"
echo ""
echo "Old model (random-walk eval): r = -0.652"
echo "Old model (seeking-captain eval): r = -0.912"
echo ""
echo "New model results:"
python << 'PYEOF'
import json
with open("eval_seeking_captain_final.json") as f:
    results = json.load(f)
    print(f"Mean correlation: {results['mean_correlation']:.3f}")
    print(f"Interpretation: {results['interpretation']}")
    for seed, data in results['results_by_seed'].items():
        print(f"  Seed {seed}: r={data['correlation']:.3f}, mean_pred={data['mean_prediction']:.4f}")
PYEOF
