import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_data, time_split_3way


def main():
    df = load_data()
    X_tr, y_tr, X_val, y_val, X_te, y_te = time_split_3way(df)
    # The test set (X_te, y_te) is deliberately not used in this script.
    X_tr = X_tr.drop(columns="Time")
    X_val = X_val.drop(columns="Time")

    # Baseline on the validation set, for a fair comparison
    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000),
    )
    lr.fit(X_tr, y_tr)
    lr_pr = average_precision_score(y_val, lr.predict_proba(X_val)[:, 1])
    print(f"Logistic Regression (validation): PR-AUC = {lr_pr:.4f}\n")

    results = []
    for num_leaves in [16, 32]:
        for min_child in [20, 100]:
            for lr_rate in [0.02]:
                model = lgb.LGBMClassifier(
                    n_estimators=3000,
                    learning_rate=lr_rate,
                    num_leaves=num_leaves,
                    min_child_samples=min_child,
                    subsample=0.8,
                    subsample_freq=1,
                    colsample_bytree=0.8,
                    random_state=42, metric="None",
                    verbose=-1,
                )
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_val, y_val)],
                    eval_metric="average_precision",
                    callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)],
                )
                pr = model.best_score_["valid_0"]["average_precision"]
                results.append((pr, num_leaves, min_child, lr_rate, model.best_iteration_))
                print(
                    f"leaves={num_leaves:<2} min_child={min_child:<3} lr={lr_rate:<4} "
                    f"best_trees={model.best_iteration_:<4} val PR-AUC={pr:.4f}"
                )

    results.sort(reverse=True)
    print("\nTop 3 settings by validation PR-AUC:")
    for r in results[:3]:
        print(f"  PR-AUC={r[0]:.4f} leaves={r[1]} min_child={r[2]} lr={r[3]} trees={r[4]}")


if __name__ == "__main__":
    main()

