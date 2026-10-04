import numpy as np
import matplotlib

matplotlib.use("Agg")  # save plots to files, no window needed
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap

from src.data import load_data, time_split_3way


def to_2d(sv):
    """Return SHAP values as (rows, features) for the fraud class, across shap versions."""
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv


def explain_one(title, x_row, sv_row, cols, score):
    print(f"{title} (model score = {score:.3f})")
    for i in np.argsort(-np.abs(sv_row))[:5]:
        print(f"  {cols[i]:<8} value={x_row[i]:>9.3f}  contribution={sv_row[i]:+.3f}")
    print()


def main():
    df = load_data()
    X_tr, y_tr, X_val, y_val, X_te, y_te = time_split_3way(df)
    X_tr = X_tr.drop(columns="Time")
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

    # Sample: all test frauds plus 2000 random normal transactions
    rng = np.random.default_rng(42)
    y = np.asarray(y_te)
    fraud_idx = np.where(y == 1)[0]
    normal_idx = rng.choice(np.where(y == 0)[0], size=2000, replace=False)
    idx = np.concatenate([fraud_idx, normal_idx])
    X_s = X_te.iloc[idx]
    y_s = y[idx]
    scores = gbm.predict_proba(X_s)[:, 1]

    explainer = shap.TreeExplainer(gbm)
    sv = to_2d(explainer.shap_values(X_s))
    cols = list(X_s.columns)

    print("Top 10 features by mean |SHAP| (sample is fraud-enriched, so this overweights fraud):")
    mean_abs = np.abs(sv).mean(axis=0)
    for i in np.argsort(-mean_abs)[:10]:
        print(f"  {cols[i]:<8} {mean_abs[i]:.3f}")
    print()

    plt.figure()
    shap.summary_plot(sv, X_s, show=False, max_display=15)
    plt.savefig("shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved shap_summary.png\n")

    # Local explanations: best-caught fraud and worst-missed fraud
    f = np.where(y_s == 1)[0]
    hi = f[np.argmax(scores[f])]
    lo = f[np.argmin(scores[f])]
    explain_one("Highest-scored fraud", X_s.iloc[hi].values, sv[hi], cols, scores[hi])
    explain_one("Lowest-scored fraud (missed)", X_s.iloc[lo].values, sv[lo], cols, scores[lo])


if __name__ == "__main__":
    main()