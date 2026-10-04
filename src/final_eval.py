import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as ap
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_data, time_split_3way


def bootstrap(y, s1, s2, n=500, seed=42):
    """Resample the test set n times and recompute PR-AUC for both models."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    a, b = [], []
    for _ in range(n):
        i = rng.choice(len(y), size=len(y), replace=True)
        if y[i].sum() == 0:
            continue
        a.append(ap(y[i], s1[i]))
        b.append(ap(y[i], s2[i]))
    return np.array(a), np.array(b)


def main():
    df = load_data()
    X_tr, y_tr, X_val, y_val, X_te, y_te = time_split_3way(df)
    X_tr = X_tr.drop(columns="Time")
    X_te = X_te.drop(columns="Time")

    # Settings were fixed using the validation set. No changes after this point.
    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000),
    )
    lr.fit(X_tr, y_tr)

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

    s_lr = lr.predict_proba(X_te)[:, 1]
    s_gbm = gbm.predict_proba(X_te)[:, 1]

    print(f"TEST frauds: {int(y_te.sum())} of {len(y_te)}")
    print(f"Logistic Regression test PR-AUC: {ap(y_te, s_lr):.4f}")
    print(f"LightGBM            test PR-AUC: {ap(y_te, s_gbm):.4f}")

    b_lr, b_gbm = bootstrap(y_te, s_lr, s_gbm)
    for name, b in [("Logistic Regression", b_lr), ("LightGBM", b_gbm)]:
        lo, hi = np.percentile(b, [2.5, 97.5])
        print(f"{name} 95% bootstrap interval: {lo:.3f} to {hi:.3f}")
    d = b_gbm - b_lr
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"LightGBM minus LogReg: mean {d.mean():+.3f}, 95% interval {lo:+.3f} to {hi:+.3f}")


if __name__ == "__main__":
    main()