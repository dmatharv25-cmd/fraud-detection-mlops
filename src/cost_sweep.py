import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_data, time_split_3way

COST_FP = 5  # cost of a false alarm (assumed, kept fixed)
COST_FN_VALUES = [10, 50, 100, 500]  # cost of a missed fraud, varied
GRID = np.linspace(0.01, 0.99, 99)


def counts(y, s, t):
    """Flag a transaction if score >= t. Return missed frauds and false alarms."""
    flag = s >= t
    fn = int(((y == 1) & ~flag).sum())
    fp = int(((y == 0) & flag).sum())
    return fn, fp


def best_threshold(y, s, cost_fn):
    costs = []
    for t in GRID:
        fn, fp = counts(y, s, t)
        costs.append(cost_fn * fn + COST_FP * fp)
    return GRID[int(np.argmin(costs))]


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

    scores = {}
    for name, model in [("LogReg", lr), ("LightGBM", gbm)]:
        scores[name] = (model.predict_proba(X_val)[:, 1], model.predict_proba(X_te)[:, 1])

    print(f"False alarm cost fixed at {COST_FP}. Thresholds chosen on validation only.\n")
    print(f"{'missed cost':<12}{'model':<10}{'thr':<6}{'test cost':<10}{'missed':<8}{'false alarms':<13}{'flag-nothing':<12}")
    for cfn in COST_FN_VALUES:
        for name, (s_val, s_te) in scores.items():
            t = best_threshold(y_val, s_val, cfn)
            fn, fp = counts(y_te, s_te, t)
            test_cost = cfn * fn + COST_FP * fp
            baseline = cfn * int(y_te.sum())
            print(f"{cfn:<12}{name:<10}{t:<6.2f}{test_cost:<10}{fn:<8}{fp:<13}{baseline:<12}")
        print()


if __name__ == "__main__":
    main()