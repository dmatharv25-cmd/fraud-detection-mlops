from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "creditcard.csv"


def load_data(path=DATA_PATH):
    """Load the dataset and sort it by time."""
    df = pd.read_csv(path)
    return df.sort_values("Time").reset_index(drop=True)


def _xy(part):
    return part.drop(columns="Class"), part["Class"]


def time_split(df, train_frac=0.8):
    """Time-based split: the first 80% of time is train, the last 20% is test."""
    cut = int(len(df) * train_frac)
    X_train, y_train = _xy(df.iloc[:cut])
    X_test, y_test = _xy(df.iloc[cut:])
    return X_train, y_train, X_test, y_test


def time_split_3way(df, train_frac=0.6, val_frac=0.2):
    """Train (first 60%), validation (next 20%), test (last 20%), all in time order."""
    a = int(len(df) * train_frac)
    b = int(len(df) * (train_frac + val_frac))
    X_train, y_train = _xy(df.iloc[:a])
    X_val, y_val = _xy(df.iloc[a:b])
    X_test, y_test = _xy(df.iloc[b:])
    return X_train, y_train, X_val, y_val, X_test, y_test