# Fraud Detection MLOps

Credit card fraud detection on the ULB dataset (284,807 transactions, 492 frauds, 0.17%), built with time-based splits and honest evaluation.

Status: in progress (Week 1 done: baseline and LightGBM comparison).

## Setup

- Train: first 60% of time (360 frauds)
- Validation: next 20% (57 frauds), used for all tuning
- Test: last 20% (75 frauds), evaluated once at the end

## Results

| Model | Validation PR-AUC | Test PR-AUC | Test 95% bootstrap interval |
|---|---|---|---|
| Logistic Regression (balanced) | 0.7725 | 0.7438 | 0.626 to 0.849 |
| LightGBM (32 leaves, lr 0.02, 1564 trees) | 0.7882 | 0.8106 | 0.719 to 0.881 |

LightGBM minus Logistic Regression on test: +0.061, 95% interval +0.006 to +0.135.

## Findings and caveats

- Accuracy is useless here (always predicting "not fraud" gives 99.8%), so PR-AUC is the metric.
- LightGBM scored higher on test and the difference interval excludes zero, but the lower end is only +0.006, so the evidence is modest. On validation the two models were nearly tied.
- Only 75 test frauds, so all scores are noisy. The bootstrap captures this noise but not drift in fraud patterns over time.
- Class weighting (scale_pos_weight) hurt LightGBM badly in my experiments. I did not establish why.
- The test set was used once. Models were tuned only on validation.

## Next

Cost-sensitive threshold, SHAP explanations, MLflow tracking, FastAPI service, drift monitoring.
