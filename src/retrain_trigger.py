import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data import load_data, time_split_3way
from src.drift import psi

MODELS = Path(__file__).resolve().parent.parent / "models"

# These limits are my choices, not standards.
PSI_LIMIT = 0.25       # a feature counts as drifted above this PSI
MAX_DRIFTED = 5        # drift condition: more than this many drifted features
MAX_CATCH_DROP = 0.10  # performance condition: catch rate falls by more than this


def catch_rate(y, s, thr):
    flagged = s >= thr
    return int(((y == 1) & flagged).sum()) / max(int(y.sum()), 1)


def main():
    force = "--force" in sys.argv
    meta = json.loads((MODELS / "meta.json").read_text())
    features, thr = meta["features"], meta["threshold"]
    model = joblib.load(MODELS / "lgbm.pkl")

    df = load_data()
    X_tr, y_tr, X_val, y_val, X_te, y_te = time_split_3way(df)
    X_tr, X_val, X_te = X_tr[features], X_val[features], X_te[features]
    y_tr, y_val, y_te = np.asarray(y_tr), np.asarray(y_val), np.asarray(y_te)

    # The test period plays the role of the new live window.
    drifted = [f for f in features if psi(X_tr[f].values, X_te[f].values) > PSI_LIMIT]
    drift_fired = len(drifted) > MAX_DRIFTED

    ref_catch = catch_rate(y_val, model.predict_proba(X_val)[:, 1], thr)
    new_catch = catch_rate(y_te, model.predict_proba(X_te)[:, 1], thr)
    perf_fired = (ref_catch - new_catch) > MAX_CATCH_DROP

    print(f"Drift check: {len(drifted)} of {len(features)} features with PSI > {PSI_LIMIT} (limit: more than {MAX_DRIFTED})")
    print(f"  -> {'FIRED' if drift_fired else 'not fired'}")
    print(f"Performance check: catch rate {ref_catch:.1%} (validation) vs {new_catch:.1%} (new window), allowed drop {MAX_CATCH_DROP:.0%}")
    print(f"  -> {'FIRED' if perf_fired else 'not fired'}")

    if not ((drift_fired and perf_fired) or force):
        print("Decision: do not retrain (both conditions are needed).")
        return

    if force:
        print("Decision: retraining because --force was given.")
    else:
        print("Decision: retraining, both conditions fired.")

    X_all = pd.concat([X_tr, X_val, X_te])
    y_all = np.concatenate([y_tr, y_val, y_te])
    candidate = lgb.LGBMClassifier(
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
    candidate.fit(X_all, y_all)
    joblib.dump(candidate, MODELS / "candidate.pkl")
    print("Saved models/candidate.pkl. The current model was not overwritten.")
    print("The candidate was trained on all data, so it cannot be evaluated here.")
    print("Promoting it needs a fresh later time window.")


if __name__ == "__main__":
    main()