"""Grid search RF hyperparameters on validation set, then evaluate best on test."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import argparse

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import ParameterGrid

from common import (
    FEATURE_COLUMNS,
    RESULTS_DIR,
    TARGET_COLUMN,
    load_features_and_split,
)


@dataclass
class Score:
    acc: float
    macro_f1: float
    bal_acc: float


def _eval(model: RandomForestClassifier, X: pd.DataFrame, y: pd.Series) -> Score:
    pred = model.predict(X)
    return Score(
        acc=accuracy_score(y, pred),
        macro_f1=f1_score(y, pred, average="macro", zero_division=0),
        bal_acc=balanced_accuracy_score(y, pred),
    )


def build_param_grid(mode: str) -> dict[str, list[Any]]:
    if mode == "fast":
        return {
            "n_estimators": [100],
            "max_depth": [4, 6, 8],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt"],
            "class_weight": [None, "balanced"],
            "random_state": [42],
            "n_jobs": [-1],
        }
    return {
        "n_estimators": [200, 400],
        "max_depth": [4, 6, 8, None],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", "log2"],
        "class_weight": [None, "balanced", "balanced_subsample"],
        "random_state": [42],
        "n_jobs": [-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fast", "slow"], default="slow")
    args = parser.parse_args()
    train_df, val_df, test_df = load_features_and_split()
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[TARGET_COLUMN]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    param_grid = build_param_grid(args.mode)

    rows: list[dict[str, Any]] = []
    best_key = (-1.0, -1.0, -1.0)
    best_params: dict[str, Any] | None = None
    best_model: RandomForestClassifier | None = None

    for params in ParameterGrid(param_grid):
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        score = _eval(model, X_val, y_val)
        row = {**params, "val_acc": score.acc, "val_macro_f1": score.macro_f1, "val_bal_acc": score.bal_acc}
        rows.append(row)

        key = (score.macro_f1, score.bal_acc, score.acc)
        if key > best_key:
            best_key = key
            best_params = params
            best_model = model

    if not rows or best_params is None or best_model is None:
        raise RuntimeError("Grid search produced no results.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / f"rf_grid_search_{args.mode}.csv"
    pd.DataFrame(rows).sort_values(
        ["val_macro_f1", "val_bal_acc", "val_acc"],
        ascending=False,
    ).to_csv(report_path, index=False)
    print(f"Saved {report_path}")

    # Refit best on train+val, then evaluate on test.
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    X_train_val = train_val_df[FEATURE_COLUMNS]
    y_train_val = train_val_df[TARGET_COLUMN]

    best_refit = RandomForestClassifier(**best_params)
    best_refit.fit(X_train_val, y_train_val)
    test_score = _eval(best_refit, X_test, y_test)

    pred = best_refit.predict(X_test)
    out = test_df[["date"]].copy()
    out["true_label"] = y_test.values
    out["pred_label"] = pred
    out_path = RESULTS_DIR / f"pred_rf_best_{args.mode}.csv"
    out.to_csv(out_path, index=False)

    print(
        "Best params:",
        {k: best_params[k] for k in sorted(best_params.keys())},
    )
    print(
        f"Test metrics: acc={test_score.acc:.4f}, "
        f"macro_f1={test_score.macro_f1:.4f}, "
        f"bal_acc={test_score.bal_acc:.4f}"
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
