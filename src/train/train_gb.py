"""Train Histogram Gradient Boosting model for 3-class trend prediction."""

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

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
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        random_state=42,
    )
    model.fit(X_train, y_train)
    pred_label = model.predict(X_test)

    out = test_df[["date"]].copy()
    out["true_label"] = y_test.values
    out["pred_label"] = pred_label

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "pred_gb.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
