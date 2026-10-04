import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_data, time_split_3way

COST_FN = 100  # cost of a missed fraud (assumed)
COST_FP = 5    # cost of a false alarm (assumed)


def cost(y, s, t):
    """Flag a transaction if score >= t. Return total cost, missed frauds, false alarms."""
    flag = s >= t
    fn = int(((y == 1) & ~flag).sum())
    fp = int(((y == 0) & flag).sum())
    return COST_FN * fn + COST_FP * fp, fn, fp


def best_threshold(y, s):
    grid = np.linspace(0.01, 0.99, 99)
    costs = [cost(y, s, t)[0] for t in grid]
    return grid[int(np.argmin(costs))]


def main():
    df = load_data()
    X_tr, y_tr, X_val, y_val, X_te, y_te = time_split_3way(df)
    X_tr, X_val, X_te = (x.drop(columns="Time") for x in (X_tr, X_val, X_te))
    y_val, y_te = np.asarray(y_val), np.asarray(y_te)

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

    print(f"Assumed costs: missed fraud = {COST_FN}, false alarm = {COST_FP}")
    print(f"Test, flag nothing:    cost = {COST_FN * int(y_te.sum())}")
    print(f"Test, flag everything: cost = {COST_FP * int((y_te == 0).sum())}\n")

    for name, model in [("Logistic Regression", lr), ("LightGBM", gbm)]:
        s_val = model.predict_proba(X_val)[:, 1]
        s_te = model.predict_proba(X_te)[:, 1]
        t = best_threshold(y_val, s_val)  # chosen on validation only
        c_val, fn_v, fp_v = cost(y_val, s_val, t)
        c_te, fn_t, fp_t = cost(y_te, s_te, t)
        print(f"{name}: threshold chosen on validation = {t:.2f}")
        print(f"  validation: cost={c_val}  missed={fn_v}/{int(y_val.sum())}  false alarms={fp_v}")
        print(f"  test:       cost={c_te}  missed={fn_t}/{int(y_te.sum())}  false alarms={fp_t}\n")


if __name__ == "__main__":
    main()