"""Train tree-based models (Random Forest) for 3-class trend prediction."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from common import (
    FEATURE_COLUMNS,
    RESULTS_DIR,
    TARGET_COLUMN,
    load_features_and_split,
)


# Step 1: multiple RF configs (max_depth, min_samples_leaf)
RF_CONFIGS = [
    {"name": "RF1", "max_depth": 3, "min_samples_leaf": 1},
    {"name": "RF2", "max_depth": 5, "min_samples_leaf": 1},
    {"name": "RF3", "max_depth": 8, "min_samples_leaf": 1},
    {"name": "RF4", "max_depth": 8, "min_samples_leaf": 5},
]


def train_random_forest(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    random_state: int = 42,
    class_weight: str = "balanced",
    n_estimators: int = 100,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
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
            train_df, val_df, test_df,
            max_depth=cfg["max_depth"],
            min_samples_leaf=cfg["min_samples_leaf"],
        )
        out_path = RESULTS_DIR / f"pred_{name.lower()}.csv"
        pred_df.to_csv(out_path, index=False)
        print(f"Saved {out_path} ({name}: max_depth={cfg['max_depth']}, min_samples_leaf={cfg['min_samples_leaf']})")


if __name__ == "__main__":
    main()
