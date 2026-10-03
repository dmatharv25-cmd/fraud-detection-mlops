from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "creditcard.csv"


def load_data(path=DATA_PATH):
    """Load the dataset and sort it by time."""
    df = pd.read_csv(path)
    return df.sort_values("Time").reset_index(drop=True)


def time_split(df, train_frac=0.8):
    """Time-based split: the first 80% of time is train, the last 20% is test."""
    cut = int(len(df) * train_frac)
    train, test = df.iloc[:cut], df.iloc[cut:]
    X_train, y_train = train.drop(columns="Class"), train["Class"]
    X_test, y_test = test.drop(columns="Class"), test["Class"]
    return X_train, y_train, X_test, y_test