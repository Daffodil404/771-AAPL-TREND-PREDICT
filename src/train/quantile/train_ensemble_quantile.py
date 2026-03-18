"""Ensemble Logistic Regression and Linear SVM for quantile labels via probability blending."""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .common_quantile import (
    FEATURE_COLUMNS,
    RESULTS_DIR,
    TARGET_COLUMN,
    load_features_and_split,
)


def align_proba(proba: np.ndarray, from_classes, to_classes) -> np.ndarray:
    """Reorder probability columns from from_classes to match to_classes."""
    index = {c: i for i, c in enumerate(from_classes)}
    return np.column_stack([proba[:, index[c]] for c in to_classes])


def main() -> None:
    train_df, val_df, test_df = load_features_and_split()
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression
    logreg = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
    )
    logreg.fit(X_train_scaled, y_train)
    logreg_prob = logreg.predict_proba(X_test_scaled)

    # Linear SVM + calibration
    svm = LinearSVC(C=0.1, class_weight="balanced", random_state=42, max_iter=5000)
    svm.fit(X_train_scaled, y_train)
    svm_cal = CalibratedClassifierCV(svm, method="sigmoid", cv="prefit")
    svm_cal.fit(X_val_scaled, y_val)
    svm_prob = svm_cal.predict_proba(X_test_scaled)

    classes = list(logreg.classes_)
    if list(svm_cal.classes_) != classes:
        svm_prob = align_proba(svm_prob, svm_cal.classes_, classes)

    prob = 0.6 * logreg_prob + 0.4 * svm_prob
    pred_idx = prob.argmax(axis=1)
    pred_label = np.array(classes)[pred_idx]

    pred_df = pd.DataFrame(
        {
            "date": test_df["date"],
            "true_label": y_test.values,
            "pred_label": pred_label,
        }
    )
    out_path = RESULTS_DIR / "pred_ensemble.csv"
    pred_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

