import mlflow
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as ap
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_data, time_split_3way

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("fraud-detection")


def log_run(name, model, params, X_tr, y_tr, X_val, y_val, X_te, y_te):
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        model.fit(X_tr, y_tr)
        val_pr = ap(y_val, model.predict_proba(X_val)[:, 1])
        test_pr = ap(y_te, model.predict_proba(X_te)[:, 1])
        mlflow.log_metric("val_pr_auc", val_pr)
        mlflow.log_metric("test_pr_auc", test_pr)
        print(f"{name}: val PR-AUC = {val_pr:.4f}, test PR-AUC = {test_pr:.4f}")


def main():
    df = load_data()
    X_tr, y_tr, X_val, y_val, X_te, y_te = time_split_3way(df)
    X_tr, X_val, X_te = (x.drop(columns="Time") for x in (X_tr, X_val, X_te))
    data = (X_tr, y_tr, X_val, y_val, X_te, y_te)

    lr_params = {"model": "LogisticRegression", "class_weight": "balanced", "max_iter": 1000}
    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000),
    )
    log_run("LogisticRegression", lr, lr_params, *data)

    gbm_params = {
        "model": "LightGBM",
        "n_estimators": 1564,
        "learning_rate": 0.02,
        "num_leaves": 32,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }
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
    log_run("LightGBM", gbm, gbm_params, *data)


if __name__ == "__main__":
    main()