"""Validation-based search for HistGradientBoostingClassifier, then test best."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import argparse

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
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


def _eval(model: HistGradientBoostingClassifier, X: pd.DataFrame, y: pd.Series) -> Score:
    pred = model.predict(X)
    return Score(
        acc=accuracy_score(y, pred),
        macro_f1=f1_score(y, pred, average="macro", zero_division=0),
        bal_acc=balanced_accuracy_score(y, pred),
    )


def build_param_grid(mode: str) -> dict[str, list[Any]]:
    # Custom weights based on provided class proportions:
    # up 0.414, down 0.383, flat 0.202
    # Base inverse-frequency (normalized) ~ down 0.79, up 0.73, flat 1.49
    # Slightly up-weight "up" to encourage more up predictions.
    custom_weights = [
        {"down": 0.79, "flat": 1.49, "up": 0.73},
        {"down": 0.80, "flat": 1.50, "up": 1.10},
        {"down": 0.75, "flat": 1.45, "up": 1.30},
    ]
    if mode == "fast":
        return {
            "learning_rate": [0.03, 0.07],
            "max_depth": [3, 5],
            "max_iter": [200],
            "min_samples_leaf": [20, 50],
            "l2_regularization": [0.0, 1.0],
            "class_weight": [None, "balanced", *custom_weights],
            "random_state": [42],
        }
    return {
        "learning_rate": [0.02, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "max_iter": [200, 400],
        "min_samples_leaf": [10, 20, 50],
        "l2_regularization": [0.0, 1.0, 5.0],
        "class_weight": [None, "balanced", *custom_weights],
        "random_state": [42],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fast", "slow"], default="fast")
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

    for params in ParameterGrid(param_grid):
        params = dict(params)
        class_weight = params.pop("class_weight")
        model = HistGradientBoostingClassifier(class_weight=class_weight, **params)
        sample_weight = None
        if isinstance(class_weight, dict):
            sample_weight = y_train.map(class_weight).values
            model = HistGradientBoostingClassifier(class_weight=None, **params)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        score = _eval(model, X_val, y_val)
        rows.append(
            {
                **params,
                "class_weight": class_weight,
                "val_acc": score.acc,
                "val_macro_f1": score.macro_f1,
                "val_bal_acc": score.bal_acc,
            }
        )
        key = (score.macro_f1, score.bal_acc, score.acc)
        if key > best_key:
            best_key = key
            best_params = {**params, "class_weight": class_weight}

    if not rows or best_params is None:
        raise RuntimeError("Grid search produced no results.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / f"hgb_grid_search_{args.mode}.csv"
    pd.DataFrame(rows).sort_values(
        ["val_macro_f1", "val_bal_acc", "val_acc"],
        ascending=False,
    ).to_csv(report_path, index=False)
    print(f"Saved {report_path}")

    # Refit best on train+val, then evaluate on test.
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    X_train_val = train_val_df[FEATURE_COLUMNS]
    y_train_val = train_val_df[TARGET_COLUMN]

    class_weight = best_params.pop("class_weight")
    best_refit = HistGradientBoostingClassifier(class_weight=class_weight, **best_params)
    sample_weight = None
    if isinstance(class_weight, dict):
        sample_weight = y_train_val.map(class_weight).values
        best_refit = HistGradientBoostingClassifier(class_weight=None, **best_params)
    best_refit.fit(X_train_val, y_train_val, sample_weight=sample_weight)
    test_score = _eval(best_refit, X_test, y_test)

    pred = best_refit.predict(X_test)
    out = test_df[["date"]].copy()
    out["true_label"] = y_test.values
    out["pred_label"] = pred
    out_path = RESULTS_DIR / f"pred_hgb_best_{args.mode}.csv"
    out.to_csv(out_path, index=False)

    display_params = {**best_params, "class_weight": class_weight}
    print("Best params:", {k: display_params[k] for k in sorted(display_params.keys())})
    print(
        f"Test metrics: acc={test_score.acc:.4f}, "
        f"macro_f1={test_score.macro_f1:.4f}, "
        f"bal_acc={test_score.bal_acc:.4f}"
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
