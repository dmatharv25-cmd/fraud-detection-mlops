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

## Cost-sensitive threshold results

Costs are assumptions, not real bank data. A false alarm costs 5, and a missed fraud costs the value in the first column. Thresholds were chosen on validation only, then applied once to the test set.

| Missed-fraud cost | LogReg test cost | LightGBM test cost | Flag nothing |
|---|---|---|---|
| 10 | 435 | 230 | 750 |
| 50 | 1075 | 985 | 3750 |
| 100 | 1745 | 1935 | 7500 |
| 500 | 6680 | 9535 | 37500 |

- Both models beat flagging nothing at every cost ratio.
- The lower-cost model flipped between a missed-fraud cost of 50 and 100 (LightGBM lower at 10 and 50, Logistic Regression lower at 100 and 500).
- The flip is not established. The gaps come from a handful of transactions, thresholds were picked on only 57 validation frauds, and I did not compute bootstrap intervals for these costs.
- Logistic Regression's best threshold hit the edge of my search grid (0.99) at low costs, so the true best cutoff may be higher.

## Explainability (SHAP)

SHAP values were computed for the final LightGBM on a test-set sample: all 75 frauds plus 2,000 random normal transactions. Values are in log-odds units, where positive pushes the score toward fraud.

![SHAP summary](shap_summary.png)

| Rank | Feature | Mean abs SHAP |
|---|---|---|
| 1 | V14 | 0.860 |
| 2 | V4 | 0.677 |
| 3 | V8 | 0.578 |
| 4 | V12 | 0.531 |
| 5 | V11 | 0.398 |

- The highest-scored fraud (score 1.000) had several extreme feature values agreeing, led by V14 (+7.3), V12, V17, V4 and V10.
- The lowest-scored fraud (score 0.000, missed) had no strong signal on the features the model relies on. Its largest contribution was only +1.2 from V14, and some features pushed the other way.
- V1 to V28 are anonymized PCA components, so SHAP shows which components drive the model but not what they mean in real life.
- The sample is fraud-enriched, so the ranking reflects what drives fraud calls, not importance across normal traffic.
- I looked at only two individual transactions. I have not checked whether the missed frauds share a pattern.

## Next

MLflow tracking, FastAPI service, drift monitoring.
