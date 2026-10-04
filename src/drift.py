import json
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import ks_2samp

from src.data import load_data, time_split_3way

MODELS = Path(__file__).resolve().parent.parent / "models"


def psi(ref, new, bins=10):
    """Population Stability Index. Bin edges come from reference quantiles."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    n = np.histogram(new, edges)[0] / len(new)
    r, n = np.clip(r, 1e-4, None), np.clip(n, 1e-4, None)
    return float(np.sum((n - r) * np.log(n / r)))


def compare(name, ref, new, features, top=5):
    rows = []
    for f in features:
        p = psi(ref[f].values, new[f].values)
        ks = ks_2samp(ref[f].values, new[f].values)
        rows.append((p, ks.statistic, f))
    rows.sort(reverse=True)
    big = sum(1 for r in rows if r[0] > 0.25)
    mid = sum(1 for r in rows if 0.1 <= r[0] <= 0.25)
    print(f"{name}: {big} features with PSI > 0.25, {mid} with PSI 0.1 to 0.25 (of {len(rows)})")
    for p, ks, f in rows[:top]:
        print(f"  {f:<8} PSI={p:.3f}  KS statistic={ks:.3f}")
    print()


def main():
    meta = json.loads((MODELS / "meta.json").read_text())
    features = meta["features"]
    model = joblib.load(MODELS / "lgbm.pkl")

    df = load_data()
    X_tr, y_tr, X_val, y_val, X_te, y_te = time_split_3way(df)
    parts = {
        "train": (X_tr[features], np.asarray(y_tr)),
        "validation": (X_val[features], np.asarray(y_val)),
        "test": (X_te[features], np.asarray(y_te)),
    }

    print("Per-period summary (reference = train):")
    for name, (X, y) in parts.items():
        s = model.predict_proba(X)[:, 1]
        print(f"  {name:<11} fraud rate={y.mean() * 100:.3f}%  mean score={s.mean():.5f}  flagged={(s >= meta['threshold']).mean() * 100:.3f}%")
    print()

    ref = parts["train"][0]
    compare("train vs validation", ref, parts["validation"][0], features)
    compare("train vs test", ref, parts["test"][0], features)


if __name__ == "__main__":
    main()