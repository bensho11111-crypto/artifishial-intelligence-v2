# Proximity Correlation Metric

## Overview

**Proximity Correlation** directly measures whether the model's catch predictions correlate with actual fish proximity, independent of captain performance.

This metric answers: *"When the model predicts high catch probability, are we actually close to fish?"*

## The Metric

### Definition

For a test episode:
1. At each timestep `t`, record:
   - Model's predicted catch probability: `sum(P[species] for all species)`
   - Actual distance to nearest fish school: `min(dist to each school)`

2. Compute **Pearson correlation** between predictions and inverse-distance (closer fish = higher inverse distance)

3. Report correlation `r ∈ [-1, 1]`:
   - `r ≈ 1.0` : Perfect — higher predictions when closer to fish
   - `r ≈ 0.0` : Random — no relationship
   - `r < 0.0` : Inverted — higher predictions when farther from fish

### Why This Matters

**Captain Score (current metric) only measures outcome:**
- "Did the boat catch fish?" (yes/no)
- Limited by captain strategy, randomness, boat dynamics
- Doesn't isolate model quality

**Proximity Correlation isolates model quality:**
- Measures pure prediction ability
- Independent of how well the captain uses predictions
- Detects if model learned fish detection at all
- Shows *when* the model is wrong (too optimistic far from fish?)

## Interpretation

### By Correlation Value

| r | Interpretation | What it means |
|---|---|---|
| **0.8 - 1.0** | Excellent | Model is learning! Predictions closely track fish proximity |
| **0.6 - 0.8** | Strong | Model has learned good signal; some noise/randomness |
| **0.4 - 0.6** | Moderate | Model has partial signal; lots of room for improvement |
| **0.2 - 0.4** | Weak | Model barely better than random; needs more training |
| **0.0 - 0.2** | None/Random | Model shows no correlation; likely untrained or architecture issue |
| **< 0.0** | Inverted | Model predicts opposite of reality — critical bug |

### By Distance Bin

The metric also groups predictions by distance to nearest school:

```
0-50m       ███████ 0.65  ← Should be highest (we're in/near school)
50-100m     █████   0.42
100-200m    ███     0.28
200-400m    ██      0.15
400m+       █       0.08  ← Should be lowest (far from fish)
```

If bars are increasing (left to right), model is working. If flat or inverted, model hasn't learned.

## Implementation

### Command Line

```bash
python tools/eval_proximity.py \
    --model checkpoints_simple_cpu/best.pt \
    --duration 300 \
    --seeds 0,1,2
```

### Output Format

```json
{
  "model": "checkpoints_simple_cpu/best.pt",
  "duration_s": 300,
  "mean_correlation": 0.42,
  "std_correlation": 0.08,
  "results_by_seed": {
    "0": {
      "correlation": 0.45,
      "p_value": 0.0001,
      "mean_pred_by_distance": {
        "0-50m": 0.65,
        "50-100m": 0.42,
        "100-200m": 0.28,
        "200-400m": 0.15,
        "400+m": 0.08
      },
      "mean_prediction": 0.31,
      "mean_distance_to_nearest_school": 187.4
    },
    ...
  }
}
```

## Comparison: Current Metrics

### Three Evaluation Metrics Now Available

| Metric | Measures | Interpretation |
|--------|----------|---|
| **Captain Score** | Catches / Random | "How good is the full system?" (model + boat dynamics + captain) |
| **Oracle Fraction** | Catches / Oracle | "How much value vs. perfect knowledge?" |
| **Proximity Correlation** | Pred ↔ Distance | "Does the model learn?" (isolated, model-only) |

### Example Scenario

A model could have:
- ✅ **High Proximity Correlation (0.7)** — model learned to detect fish
- ✅ **Low Captain Score (1.1)** — but captain isn't using predictions well
- ⚠️ **Action**: Tune captain strategy, not model architecture

Conversely:
- ❌ **Low Proximity Correlation (0.1)** — model didn't learn
- ❌ **Low Captain Score (1.0)** — can't catch what model can't detect
- ⚠️ **Action**: Collect more training data, change model architecture

## Training Interpretation

### Early Training (untrained model)
- Captain Score: ~1.0 (random)
- Proximity Correlation: ~0.0 (no signal)
- → Model is random; captain can't help

### Mid Training (learning)
- Captain Score: ~2.0 (2× random)
- Proximity Correlation: ~0.4 (weak signal)
- → Model learning but noisy; captain helping slightly

### Late Training (converged)
- Captain Score: ~4.0 (4× random)
- Proximity Correlation: ~0.7 (strong signal)
- → Model learned well; captain using predictions effectively

## Statistical Notes

- **Pearson correlation** assumes linear relationship
- **p-value** shows significance (p < 0.05 is "real" correlation)
- **Multiple seeds** reduce noise; mean ± std shows stability
- **N observations** = duration_s (one per timestep)

## Future Extensions

Possible refinements:
1. **Per-species correlation** — which species does model learn best?
2. **Temporal correlation** — how quickly does model respond to approaching fish?
3. **Spatial correlation** — does model predict heading-relative distance better?
4. **Bearing correlation** — does model know *which direction* fish are?
