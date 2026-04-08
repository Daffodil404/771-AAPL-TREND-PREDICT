"""Train LightGBM model for 3-class trend prediction."""

import pandas as pd
import lightgbm as lgb

from common import (
    FEATURE_COLUMNS,
    RESULTS_DIR,
    TARGET_COLUMN,
    load_features_and_split,
)


def main() -> None:
    train_df, val_df, test_df = load_features_and_split()
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=15,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    pred_label = model.predict(X_test, num_iteration=model.best_iteration_)

    out = test_df[["date"]].copy()
    out["true_label"] = y_test.values
    out["pred_label"] = pred_label

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "pred_lgbm.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    importance = model.booster_.feature_importance(importance_type="gain")

    importance_df = pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "importance": importance}
    ).sort_values("importance", ascending=False)

    print(importance_df)


if __name__ == "__main__":
    main()
