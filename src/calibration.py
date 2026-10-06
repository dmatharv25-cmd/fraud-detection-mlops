import numpy as np
import lightgbm as lgb
from sklearn.metrics import brier_score_loss, log_loss

from src.data import load_data, time_split_3way

# Fixed score bins. Scores are extreme, so most rows fall in the first bin.
EDGES = [0.0, 0.001, 0.01, 0.11, 0.5, 0.9, 1.0001]


def report(name, y, s):
    y = np.asarray(y)
    base = y.mean()
    print(f"\n{name}: {len(y)} rows, {int(y.sum())} frauds, fraud rate {base:.4%}")
    print(f"  mean score          {s.mean():.5f}")
    print(f"  Brier score         {brier_score_loss(y, s):.6f}")
    print(f"  Brier, constant rate {brier_score_loss(y, np.full(len(y), base)):.6f}  (lower is better)")
    print(f"  log loss            {log_loss(y, np.clip(s, 1e-6, 1 - 1e-6)):.5f}")
    print(f"  {'score bin':<16}{'rows':<10}{'frauds':<9}{'mean score':<13}{'actual rate':<12}")
    idx = np.digitize(s, EDGES[1:-1])
    for k in range(len(EDGES) - 1):
        m = idx == k
        if m.sum() == 0:
            continue
        lo, hi = EDGES[k], min(EDGES[k + 1], 1.0)
        print(f"  {lo:<6g}-{hi:<9g}{int(m.sum()):<10}{int(y[m].sum()):<9}{s[m].mean():<13.4f}{y[m].mean():<12.4f}")


def main():
    df = load_data()
    X_tr, y_tr, X_val, y_val, X_te, y_te = time_split_3way(df)
    X_tr = X_tr.drop(columns="Time")
    X_val = X_val.drop(columns="Time")
    X_te = X_te.drop(columns="Time")

    gbm = lgb.LGBMClassifier(
        n_estimators=1564,
        learning_rate=0.02,
        num_leaves=32,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    gbm.fit(X_tr, y_tr)

    report("VALIDATION", y_val, gbm.predict_proba(X_val)[:, 1])
    report("TEST", y_te, gbm.predict_proba(X_te)[:, 1])


if __name__ == "__main__":
    main()