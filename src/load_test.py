import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

from src.data import load_data, time_split_3way

URL = "http://127.0.0.1:8000"
MODELS = Path(__file__).resolve().parent.parent / "models"
LEVELS = [1, 4, 16, 32]   # number of concurrent clients
N_REQUESTS = 400          # requests per level


def worker(payloads):
    session = requests.Session()
    lat, errors = [], 0
    for p in payloads:
        t0 = time.perf_counter()
        try:
            r = session.post(f"{URL}/predict", json=p, timeout=30)
            ok = r.status_code == 200
        except requests.RequestException:
            ok = False
        lat.append((time.perf_counter() - t0) * 1000)
        if not ok:
            errors += 1
    return lat, errors


def run_level(payloads, c):
    chunks = [payloads[i::c] for i in range(c)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=c) as ex:
        results = list(ex.map(worker, chunks))
    elapsed = time.perf_counter() - t0
    lat = np.array([x for r in results for x in r[0]])
    errors = sum(r[1] for r in results)
    p50, p95, p99 = np.percentile(lat, [50, 95, 99])
    print(f"{c:<9}{len(lat) / elapsed:<12.1f}{p50:<10.1f}{p95:<10.1f}{p99:<10.1f}{errors:<8}")


def main():
    meta = json.loads((MODELS / "meta.json").read_text())
    features = meta["features"]

    df = load_data()
    _, _, _, _, X_te, y_te = time_split_3way(df)
    X_te = X_te.drop(columns="Time")
    y = np.asarray(y_te)

    rng = np.random.default_rng(0)
    fraud_idx = rng.choice(np.where(y == 1)[0], size=10, replace=False)
    normal_idx = rng.choice(np.where(y == 0)[0], size=190, replace=False)
    idx = np.concatenate([fraud_idx, normal_idx])
    sample = X_te.iloc[idx][features]
    base = [sample.iloc[i].to_dict() for i in range(len(sample))]
    payloads = [base[i % len(base)] for i in range(N_REQUESTS)]

    try:
        print("Health:", requests.get(f"{URL}/health", timeout=5).json())
    except requests.RequestException:
        print("Server not reachable. Start it first: python -m uvicorn src.api:app --port 8000")
        return

    # Warm-up, not timed
    worker(payloads[:20])

    print(f"\n{N_REQUESTS} requests per level, client and server on the same machine")
    print(f"{'clients':<9}{'req/s':<12}{'p50 ms':<10}{'p95 ms':<10}{'p99 ms':<10}{'errors':<8}")
    for c in LEVELS:
        run_level(payloads, c)


if __name__ == "__main__":
    main()