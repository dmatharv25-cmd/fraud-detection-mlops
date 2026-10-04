import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score as ap

from src.data import load_data, time_split_3way

MODELS = Path(__file__).resolve().parent.parent / "models"


def boot_interval(y, s, n=500, seed=42):
    """95% bootstrap interval for PR-AUC."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.choice(len(y), size=len(y), replace=True)
        if y[i].sum() == 0:
            continue
        vals.append(ap(y[i], s[i]))
    return np.percentile(vals, [2.5, 97.5])


def main():
    meta = json.loads((MODELS / "meta.json").read_text())
    features = meta["features"]
    thr = meta["threshold"]
    model = joblib.load(MODELS / "lgbm.pkl")

    df = load_data()
    _, _, _, _, X_te, y_te = time_split_3way(df)
    X_te = X_te[features]
    y = np.asarray(y_te)
    s = model.predict_proba(X_te)[:, 1]

    half = len(y) // 2
    halves = {
        "test first half": (y[:half], s[:half]),
        "test second half": (y[half:], s[half:]),
    }

    for name, (yh, sh) in halves.items():
        lo, hi = boot_interval(yh, sh)
        flagged = sh >= thr
        caught = int(((yh == 1) & flagged).sum())
        false_alarms = int(((yh == 0) & flagged).sum())
        print(f"{name}: rows={len(yh)} frauds={int(yh.sum())} fraud rate={yh.mean() * 100:.3f}%")
        print(f"  PR-AUC={ap(yh, sh):.4f}  95% bootstrap interval {lo:.3f} to {hi:.3f}")
        print(f"  flagged={flagged.mean() * 100:.3f}%  caught={caught}/{int(yh.sum())}  false alarms={false_alarms}")
        print()


if __name__ == "__main__":
    main()