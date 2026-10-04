import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np

from src.data import load_data, time_split_3way
from src.threshold import best_threshold

MODELS = Path(__file__).resolve().parent.parent / "models"


def main():
    df = load_data()
    X_tr, y_tr, X_val, y_val, X_te, y_te = time_split_3way(df)
    X_tr = X_tr.drop(columns="Time")
    X_val = X_val.drop(columns="Time")

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

    # Cutoff chosen on validation only (missed fraud = 100, false alarm = 5)
    s_val = gbm.predict_proba(X_val)[:, 1]
    threshold = float(best_threshold(np.asarray(y_val), s_val))

    MODELS.mkdir(exist_ok=True)
    joblib.dump(gbm, MODELS / "lgbm.pkl")
    meta = {
        "features": list(X_tr.columns),
        "threshold": threshold,
        "cost_missed_fraud": 100,
        "cost_false_alarm": 5,
    }
    (MODELS / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved models/lgbm.pkl and models/meta.json. Threshold = {threshold:.2f}")
    print(f"{len(meta['features'])} features: {meta['features'][:3]} ... {meta['features'][-1]}")


if __name__ == "__main__":
    main()