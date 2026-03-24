"""Train logistic regression models for quantile-labeled trend prediction."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from src.train.quantile.common_quantile import (
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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # multinomial
    model1 = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
        multi_class="auto",
    )
    model1.fit(X_train_scaled, y_train)
    pred_label1 = model1.predict(X_test_scaled)

    # OvR
    model2 = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    )
    model2.fit(X_train_scaled, y_train)
    pred_label2 = model2.predict(X_test_scaled)

    pred_df1 = pd.DataFrame(
        {
            "date": test_df["date"],
            "true_label": y_test.values,
            "pred_label": pred_label1,
        }
    )
    pred_df2 = pd.DataFrame(
        {
            "date": test_df["date"],
            "true_label": y_test.values,
            "pred_label": pred_label2,
        }
    )
    pred_df1.to_csv(RESULTS_DIR / "pred_logreg_multinomial.csv", index=False)
    pred_df2.to_csv(RESULTS_DIR / "pred_logreg_ovr.csv", index=False)
    print(f"Saved {RESULTS_DIR / 'pred_logreg_multinomial.csv'}")
    print(f"Saved {RESULTS_DIR / 'pred_logreg_ovr.csv'}")


if __name__ == "__main__":
    main()
