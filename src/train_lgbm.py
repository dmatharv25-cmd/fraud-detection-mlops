import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_data, time_split


def main():
    df = load_data()
    X_tr, y_tr, X_te, y_te = time_split(df)
    X_tr = X_tr.drop(columns="Time")
    X_te = X_te.drop(columns="Time")

    # Fair baseline: Logistic Regression without Time
    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000),
    )
    lr.fit(X_tr, y_tr)
    lr_pr = average_precision_score(y_te, lr.predict_proba(X_te)[:, 1])
    print(f"Logistic Regression (no Time): PR-AUC = {lr_pr:.4f}")

    for w in [1, 5, 20, 100]:
        model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=50,
            scale_pos_weight=w,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_tr, y_tr)
        pr = average_precision_score(y_te, model.predict_proba(X_te)[:, 1])
        print(f"LightGBM scale_pos_weight={w}: PR-AUC = {pr:.4f}")


if __name__ == "__main__":
    main()