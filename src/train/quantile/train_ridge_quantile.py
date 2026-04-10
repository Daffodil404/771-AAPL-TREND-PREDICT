"""Train a Ridge Classifier for quantile-labeled 3-class trend prediction."""

import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

from common_quantile import (
    FEATURE_COLUMNS,
    RESULTS_DIR,
    TARGET_COLUMN,
    load_features_and_split,
)


def main() -> None:
    train_df, _, test_df = load_features_and_split()
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RidgeClassifier(alpha=1.0, class_weight="balanced", random_state=42)
    model.fit(X_train_scaled, y_train)
    pred_label = model.predict(X_test_scaled)

    pred_df = pd.DataFrame(
        {
            "date": test_df["date"],
            "true_label": y_test.values,
            "pred_label": pred_label,
        }
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "pred_ridge.csv"
    pred_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
