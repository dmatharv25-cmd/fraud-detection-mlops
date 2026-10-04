import numpy as np
import pandas as pd

from src.data import time_split_3way
from src.drift import psi
from src.threshold import cost, COST_FN, COST_FP


def make_df(n=1000):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "Time": rng.permutation(n).astype(float),
        "V1": rng.normal(size=n),
        "Class": (rng.random(n) < 0.05).astype(int),
    })
    return df.sort_values("Time").reset_index(drop=True)


def test_split_is_in_time_order_with_no_overlap():
    df = make_df()
    X_tr, y_tr, X_val, y_val, X_te, y_te = time_split_3way(df)
    assert X_tr["Time"].max() < X_val["Time"].min()
    assert X_val["Time"].max() < X_te["Time"].min()
    assert len(X_tr) + len(X_val) + len(X_te) == len(df)


def test_psi_is_zero_for_identical_data():
    x = np.random.default_rng(1).normal(size=5000)
    assert psi(x, x.copy()) < 1e-9


def test_psi_is_large_for_shifted_data():
    rng = np.random.default_rng(2)
    ref = rng.normal(size=5000)
    shifted = rng.normal(loc=2.0, size=5000)
    assert psi(ref, shifted) > 0.25


def test_cost_arithmetic():
    y = np.array([1, 1, 0, 0, 0])
    s = np.array([0.9, 0.1, 0.8, 0.2, 0.3])
    # threshold 0.5 flags indices 0 and 2: one fraud caught, one missed, one false alarm
    total, fn, fp = cost(y, s, 0.5)
    assert fn == 1
    assert fp == 1
    assert total == COST_FN * 1 + COST_FP * 1