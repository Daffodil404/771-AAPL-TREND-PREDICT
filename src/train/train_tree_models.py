"""Train tree-based models (Random Forest) for 3-class trend prediction."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from common import (
    FEATURE_COLUMNS,
    RESULTS_DIR,
    TARGET_COLUMN,
    load_features_and_split,
)


# 只保留最终报告中使用的 RF 变体：
# - RF5: 最优 RF baseline（无 class_weight，max_depth=6, min_samples_leaf=1）
# - RF6: 与 RF5 类似但更深/更大的叶子，用于对比无 class_weight 情况下的不同复杂度
RF_CONFIGS = [
    {"name": "RF5", "max_depth": 6, "min_samples_leaf": 1, "class_weight": None},
    {"name": "RF6", "max_depth": 8, "min_samples_leaf": 2, "class_weight": None},
]


def train_random_forest(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    random_state: int = 42,
    class_weight: str = None,
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    max_features: str | None = "sqrt",
) -> pd.DataFrame:
    """Fit Random Forest on train, predict on test; return pred DataFrame (date, true_label, pred_label)."""
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight=class_weight,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
    )
    model.fit(X_train, y_train)
    pred_label = model.predict(X_test)

    out = test_df[["date"]].copy()
    out["true_label"] = y_test.values
    out["pred_label"] = pred_label
    return out


def main() -> None:
    train_df, val_df, test_df = load_features_and_split()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for cfg in RF_CONFIGS:
        name = cfg["name"]
        pred_df = train_random_forest(
            train_df,
            val_df,
            test_df,
            max_depth=cfg["max_depth"],
            min_samples_leaf=cfg["min_samples_leaf"],
            class_weight=cfg["class_weight"],
        )
        out_path = RESULTS_DIR / f"pred_{name.lower()}.csv"
        pred_df.to_csv(out_path, index=False)
        print(
            f"Saved {out_path} "
            f"({name}: max_depth={cfg['max_depth']}, "
            f"min_samples_leaf={cfg['min_samples_leaf']}, "
            f"class_weight={cfg['class_weight']})"
        )


if __name__ == "__main__":
    main()
