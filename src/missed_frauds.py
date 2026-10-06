import numpy as np
import pandas as pd
import lightgbm as lgb

from src.data import load_data, time_split_3way

CUTOFF = 0.11  # the API cutoff, chosen earlier on validation


def main():
    df = load_data()
    X_tr, y_tr, X_val, y_val, X_te, y_te = time_split_3way(df)
    X_tr = X_tr.drop(columns="Time")

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

    # Scale for comparing features: std of each feature among normal training rows
    scale = X_tr[y_tr == 0].std()

    parts = []
    for name, X, y in [("validation", X_val, y_val), ("test", X_te, y_te)]:
        f = X[y == 1].copy()
        f["score"] = gbm.predict_proba(f.drop(columns="Time"))[:, 1]
        f["split"] = name
        parts.append(f)
    fr = pd.concat(parts)
    fr["missed"] = fr["score"] < CUTOFF

    caught = fr[~fr["missed"]]
    missed = fr[fr["missed"]]
    print(f"Frauds pooled: {len(fr)}  caught: {len(caught)}  missed: {len(missed)}  (cutoff {CUTOFF})")
    for s in ["validation", "test"]:
        m = fr[fr["split"] == s]
        print(f"  {s}: {int(m['missed'].sum())} missed of {len(m)}")

    print(f"\nAmount median: caught {caught['Amount'].median():.2f}  missed {missed['Amount'].median():.2f}")
    print(f"Amount mean:   caught {caught['Amount'].mean():.2f}  missed {missed['Amount'].mean():.2f}")

    feats = [c for c in X_tr.columns]
    diff = (missed[feats].mean() - caught[feats].mean()) / scale
    top = diff.reindex(diff.abs().sort_values(ascending=False).index).head(8)
    print("\nFeatures where missed and caught frauds differ most")
    print("(difference in means, in units of the normal-transaction std)")
    print(f"{'feature':<10}{'caught mean':<14}{'missed mean':<14}{'diff (std units)':<16}")
    for f_, d in top.items():
        print(f"{f_:<10}{caught[f_].mean():<14.2f}{missed[f_].mean():<14.2f}{d:<16.2f}")

    print("\nMissed frauds, lowest scores first (score, Amount, V14, V12, V4)")
    cols = ["score", "Amount", "V14", "V12", "V4"]
    print(missed.sort_values("score")[cols].head(10).round(3).to_string())


if __name__ == "__main__":
    main()