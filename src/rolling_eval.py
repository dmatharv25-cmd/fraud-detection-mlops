import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as ap
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_data

BLOCK = 6 * 3600   # 6-hour test blocks
FIRST_TEST = 4     # the first 4 blocks (24 hours) are the minimum training period


def main():
    df = load_data()
    X = df.drop(columns=["Class", "Time"])
    y = df["Class"].values
    t = df["Time"].values
    block_id = (t // BLOCK).astype(int)
    last = block_id.max()

    print(f"{'test block (hours)':<20}{'train frauds':<14}{'test frauds':<13}{'LogReg PR-AUC':<15}{'LightGBM PR-AUC':<16}")
    lr_scores, gbm_scores = [], []
    for b in range(FIRST_TEST, last + 1):
        tr = block_id < b
        te = block_id == b
        if y[te].sum() < 5:
            print(f"{b * 6}-{(b + 1) * 6:<17}skipped (fewer than 5 test frauds)")
            continue

        lr = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000),
        )
        lr.fit(X[tr], y[tr])
        gbm = lgb.LGBMClassifier(
            n_estimators=600,
            learning_rate=0.02,
            num_leaves=32,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        gbm.fit(X[tr], y[tr])

        a = ap(y[te], lr.predict_proba(X[te])[:, 1])
        g = ap(y[te], gbm.predict_proba(X[te])[:, 1])
        lr_scores.append(a)
        gbm_scores.append(g)
        label = f"{b * 6}-{(b + 1) * 6}"
        print(f"{label:<20}{int(y[tr].sum()):<14}{int(y[te].sum()):<13}{a:<15.4f}{g:<16.4f}")

    print()
    print(f"Blocks evaluated: {len(lr_scores)}")
    print(f"LogReg   mean={np.mean(lr_scores):.4f}  min={np.min(lr_scores):.4f}  max={np.max(lr_scores):.4f}")
    print(f"LightGBM mean={np.mean(gbm_scores):.4f}  min={np.min(gbm_scores):.4f}  max={np.max(gbm_scores):.4f}")
    wins = sum(g > a for a, g in zip(lr_scores, gbm_scores))
    print(f"LightGBM higher in {wins} of {len(lr_scores)} blocks")


if __name__ == "__main__":
    main()