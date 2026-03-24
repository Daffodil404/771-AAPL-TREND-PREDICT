"""Train LightGBM on quantile-labeled data."""

import pandas as pd
import lightgbm as lgb

from src.train.quantile.common_quantile import (
    RESULTS_DIR,
    TARGET_COLUMN,
    TREE_FEATURE_COLUMNS,
    load_features_and_split,
)


def main() -> None:
    train_df, val_df, test_df = load_features_and_split()
    X_train = train_df[TREE_FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[TREE_FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    label_map = {"down": 0, "flat": 1, "up": 2}
    y_train_num = y_train.map(label_map)

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        learning_rate=0.05,
        n_estimators=300,
        max_depth=6,
        random_state=42,
    )
    model.fit(X_train, y_train_num)
    pred_num = model.predict(X_test)
    inv_label_map = {v: k for k, v in label_map.items()}
    pred_label = pd.Series(pred_num).map(inv_label_map).values

    out = test_df[["date"]].copy()
    out["true_label"] = y_test.values
    out["pred_label"] = pred_label

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "pred_lgbm.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    importance = model.feature_importances_
    importance_df = pd.DataFrame(
        {"feature": TREE_FEATURE_COLUMNS, "importance": importance}
    ).sort_values("importance", ascending=False)
    print(importance_df)


if __name__ == "__main__":
    main()
