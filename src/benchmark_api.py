import json
import time
from pathlib import Path

import joblib
import numpy as np
import requests

from src.data import load_data, time_split_3way

URL = "http://127.0.0.1:8000"
MODELS = Path(__file__).resolve().parent.parent / "models"


def main():
    meta = json.loads((MODELS / "meta.json").read_text())
    features = meta["features"]
    model = joblib.load(MODELS / "lgbm.pkl")

    df = load_data()
    _, _, _, _, X_te, y_te = time_split_3way(df)
    X_te = X_te.drop(columns="Time")
    y = np.asarray(y_te)

    # 10 frauds + 190 normal transactions from the test set
    rng = np.random.default_rng(0)
    fraud_idx = rng.choice(np.where(y == 1)[0], size=10, replace=False)
    normal_idx = rng.choice(np.where(y == 0)[0], size=190, replace=False)
    idx = np.concatenate([fraud_idx, normal_idx])
    sample = X_te.iloc[idx][features]

    expected = model.predict_proba(sample)[:, 1]

    session = requests.Session()
    print("Health:", session.get(f"{URL}/health").json())

    # Warm-up requests are not timed
    for i in range(5):
        session.post(f"{URL}/predict", json=sample.iloc[i].to_dict())

    latencies, got = [], []
    for i in range(len(sample)):
        payload = sample.iloc[i].to_dict()
        t0 = time.perf_counter()
        r = session.post(f"{URL}/predict", json=payload)
        latencies.append((time.perf_counter() - t0) * 1000)
        got.append(r.json()["fraud_score"])

    got = np.array(got)
    print(f"Requests timed: {len(latencies)}")
    print(f"Max |API score - direct score|: {np.abs(got - expected).max():.2e}")
    print(f"Flagged: {int((got >= meta['threshold']).sum())} of {len(got)} (10 were real frauds)")
    print(f"Latency ms: median={np.percentile(latencies, 50):.1f}  p95={np.percentile(latencies, 95):.1f}  p99={np.percentile(latencies, 99):.1f}")

    bad = session.post(f"{URL}/predict", json={"V1": 1.0})
    print(f"Incomplete request returns status {bad.status_code} (expected 422)")


if __name__ == "__main__":
    main()