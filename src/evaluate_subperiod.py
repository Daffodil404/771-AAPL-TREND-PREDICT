"""Evaluate predictions by sub-periods within the test set."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"

PRED_FILES = {
    "logreg_multinomial": "pred_logreg_multinomial.csv",
    "logreg_ovr": "pred_logreg_ovr.csv",
    "rf_best_fast": "pred_rf_best_fast.csv",
    "rf_best_slow": "pred_rf_best_slow.csv",
    "rf1": "pred_rf1.csv",
    "rf2": "pred_rf2.csv",
    "rf3": "pred_rf3.csv",
    "rf4": "pred_rf4.csv",
    "rf5": "pred_rf5.csv",
    "rf6": "pred_rf6.csv",
    "hgb_best_fast": "pred_hgb_best_fast.csv",
    "gb": "pred_gb.csv",
}

LABELS = ["down", "flat", "up"]

PERIODS = [
    ("2016", "2016-01-14", "2016-12-31"),
    ("2017-2019_bull", "2017-01-01", "2019-12-31"),
    ("2020_covid", "2020-01-01", "2020-12-31"),
    ("2021_tech_boom", "2021-01-01", "2021-12-31"),
    ("2022_rate_hikes", "2022-01-01", "2022-12-31"),
]


def _eval_slice(df: pd.DataFrame) -> dict[str, float]:
    y_true = df["true_label"]
    y_pred = df["pred_label"]
    return {
        "acc": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
    }


def _period_filter(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))
    return df.loc[mask]


def _write_confusion(name: str, period: str, df: pd.DataFrame) -> str:
    cm = confusion_matrix(df["true_label"], df["pred_label"], labels=LABELS)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in LABELS], columns=[f"pred_{l}" for l in LABELS])
    lines = [
        f"\n=== {name} | {period} ===",
        f"n={len(df)}",
        "Confusion matrix:",
        cm_df.to_string(),
    ]
    return "\n".join(lines)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    detail_blocks: list[str] = []

    for name, filename in PRED_FILES.items():
        path = RESULTS_DIR / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        for period_name, start, end in PERIODS:
            slice_df = _period_filter(df, start, end)
            if slice_df.empty:
                continue
            metrics = _eval_slice(slice_df)
            rows.append(
                {
                    "method": name,
                    "period": period_name,
                    "n": len(slice_df),
                    "acc": metrics["acc"],
                    "macro_f1": metrics["macro_f1"],
                    "bal_acc": metrics["bal_acc"],
                }
            )
            detail_blocks.append(_write_confusion(name, period_name, slice_df))

    out_csv = REPORTS_DIR / "evaluation_subperiod.csv"
    pd.DataFrame(rows).sort_values(["period", "acc"], ascending=[True, False]).to_csv(out_csv, index=False)
    out_txt = REPORTS_DIR / "evaluation_subperiod.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(detail_blocks))

    print(f"Saved {out_csv}")
    print(f"Saved {out_txt}")


if __name__ == "__main__":
    main()
