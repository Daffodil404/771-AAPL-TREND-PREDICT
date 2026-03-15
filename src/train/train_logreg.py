"""Train a logistic regression model for 3-class trend prediction."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

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

    # 标准化：避免 volume 等大数值特征主导，且 LR 对尺度敏感
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # class_weight='balanced'：按类别比例反比加权，避免全预测多数类（up）
    # balanced：按类别样本数反比加权，通常比手设权重更稳
    model1 = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    model1.fit(X_train_scaled, y_train)
    pred_label1 = model1.predict(X_test_scaled)
    # OvR 内部是 0/1 二分类，不能传三类的 dict，用 "balanced"
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
