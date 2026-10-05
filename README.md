# Fraud Detection MLOps

[![tests](https://github.com/dmatharv25-cmd/fraud-detection-mlops/actions/workflows/tests.yml/badge.svg)](https://github.com/dmatharv25-cmd/fraud-detection-mlops/actions/workflows/tests.yml)

The badge means the 4 unit tests pass (time-split order, PSI, cost arithmetic). They check code logic, not model quality.

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

## Prediction API (FastAPI)

The final LightGBM is served with FastAPI. The cutoff (0.11) was chosen on validation using assumed costs of 100 per missed fraud and 5 per false alarm.

Run it:

```
python -m src.save_model
python -m uvicorn src.api:app --port 8000
```

- `GET /health` returns the status, the number of features and the threshold.
- `POST /predict` takes 29 numeric fields (V1 to V28 and Amount) and returns the fraud score, a flagged true/false, the threshold and the in-server scoring time. Missing or non-numeric fields get a 422 error.

Benchmark on 200 test-set transactions (10 frauds, 190 normal), sent one at a time:

| Metric | Value |
|---|---|
| Median latency | 6.8 ms |
| p95 latency | 8.6 ms |
| p99 latency | 9.2 ms |
| Max difference, API score vs direct score | 0 |

- Latency was measured from the client, including the HTTP round trip, with client and server on the same laptop and one request at a time. This is not a load test.
- The benchmark flagged 9 of 200 transactions, but it did not compare flags to labels, so that number is not a detection rate. For detection quality, use the cost-threshold section above.
- The model file is not in Git. Run `python -m src.save_model` to create it.

## Drift monitoring

Drift is measured with PSI (Population Stability Index) and the KS test, written with numpy and scipy. The reference period is the training split. The commonly used PSI cutoffs (0.1 and 0.25) are conventions, not laws.

| Period | Fraud rate | Mean model score | Flagged at 0.11 |
|---|---|---|---|
| Train | 0.211% | 0.00211 | 0.211% |
| Validation | 0.100% | 0.00078 | 0.084% |
| Test | 0.132% | 0.00107 | 0.111% |

- Against the training period, 8 of 29 features have PSI above 0.25 in validation and 7 of 29 in test.
- V1, V3, V28, V11 and V25 lead both comparisons. V1 and V3 are above 1.0 in both. PSI and KS agree on these features.
- The mean score and flag rate fall and rise together with the fraud rate. When labels arrive late, this is the early signal available in production.

Does the drift hurt the model? The test period was split into two halves by time and scored with the same saved model:

| | First half | Second half |
|---|---|---|
| Frauds | 53 | 22 |
| PR-AUC | 0.848 (95% interval 0.748 to 0.934) | 0.730 (95% interval 0.545 to 0.888) |
| Frauds caught at 0.11 | 40 of 53 (75%) | 16 of 22 (73%) |
| False alarms | 1 | 6 |

- Drift alerts fired, but the catch rate stayed about the same. An alert means look closer, not the model is broken.
- The PR-AUC intervals overlap, and I did not test the difference directly, so I cannot call the drop a real decline. Part of it may come from the fall in fraud volume, but I did not separate that from model decay.
- The data covers only about 48 hours. This demonstrates the monitoring method. It is not evidence of long-term drift, and some of the shift may be time-of-day effects.

## Retraining trigger

`python -m src.retrain_trigger` decides whether to retrain. It retrains only when both conditions fire:

- Drift: more than 5 of 29 features have PSI above 0.25 against the training period.
- Performance: the share of frauds caught at the 0.11 cutoff falls by more than 10 percentage points compared with validation.

Result on this data, with the test period as the new window:

| Check | Value | Fired |
|---|---|---|
| Drifted features | 7 of 29 | Yes |
| Catch rate, validation vs new window | 75.4% vs 74.7% | No |

- Decision: do not retrain. Drift alone is not enough, because earlier the drift alerts fired while the catch rate barely moved.
- The limits (PSI 0.25, more than 5 features, a 10 point drop) are my choices, not standards.
- The catch-rate comparison rests on 57 validation frauds and 75 test frauds, so a 10 point margin is a loose guard.
- `--force` trains a candidate on all data and saves it to `models/candidate.pkl`. It never overwrites the current model. The candidate cannot be evaluated here because no unseen data is left. Promoting it would need a fresh later time window.

## Docker

The prediction API can run in a container. The model file is not in Git, so create it first, then build and run:

```
python -m src.save_model
docker build -t fraud-api .
docker run --rm -p 8000:8000 fraud-api
```

The same benchmark (200 test-set transactions, one request at a time) was run against the local server and against the container:

| | Local server | Docker container |
|---|---|---|
| Median latency | 6.8 ms | 10.2 ms |
| p95 latency | 8.6 ms | 11.7 ms |
| p99 latency | 9.2 ms | 13.0 ms |
| Max score difference vs direct scoring | 0 | 2.6e-26 |

- Scores match to floating-point rounding. I did not verify the cause of the tiny difference.
- The container is about 3.5 ms slower at the median. I did not measure where the extra time goes.
- The model is copied in from the local `models/` folder at build time, so the image only works with that exact model. A real system would load it from a registry or storage at startup.
- The image also contains `candidate.pkl`, which the API never loads.
- Latency is one request at a time on one laptop. This is not a load test.

## Next

Load testing, a fresh-window evaluation of the retrained candidate.
