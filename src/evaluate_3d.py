"""Evaluate 3-day prediction results."""

from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "3-day-return"
REPORTS_DIR = PROJECT_ROOT / "reports" / "3-day-return"

PRED_FILES = {
    "random": "pred_random.csv",
    "momentum": "pred_momentum.csv",
    "ma": "pred_ma.csv",
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
    "gb": "pred_gb.csv",
    "hgb_best_fast": "pred_hgb_best_fast.csv",
    "lgbm": "pred_lgbm.csv",
}


def evaluate_one(pred_path: Path) -> tuple[float, pd.DataFrame, str]:
    df = pd.read_csv(pred_path)
    y_true = df["true_label"]
    y_pred = df["pred_label"]
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=["down", "flat", "up"])
    cm_df = pd.DataFrame(
        cm,
        index=["true_down", "true_flat", "true_up"],
        columns=["pred_down", "pred_flat", "pred_up"],
    )
    report_str = classification_report(y_true, y_pred, labels=["down", "flat", "up"], zero_division=0)
    return acc, cm_df, report_str


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    all_reports = []

    for name, filename in PRED_FILES.items():
        path = RESULTS_DIR / filename
        if not path.exists():
            print(f"Skip (not found): {path}")
            continue
        acc, cm_df, report_str = evaluate_one(path)
        rows.append({"method": name, "accuracy": acc})
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.4f}")
        print("Confusion matrix:")
        print(cm_df.to_string())
        print("\nClassification report (precision / recall / F1):")
        print(report_str)
        all_reports.append((name, acc, cm_df, report_str))

    acc_df = pd.DataFrame(rows)
    if not acc_df.empty:
        acc_df = acc_df.sort_values("accuracy", ascending=False)
        print("\n" + "=" * 50)
        print("Accuracy comparison")
        print(acc_df.to_string(index=False))
        acc_df.to_csv(REPORTS_DIR / "evaluation_accuracy.csv", index=False)
        print(f"\nSaved {REPORTS_DIR / 'evaluation_accuracy.csv'}")

    with open(REPORTS_DIR / "evaluation_detail.txt", "w", encoding="utf-8") as f:
        for name, acc, cm_df, report_str in all_reports:
            f.write(f"\n=== {name} ===\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write("Confusion matrix:\n")
            f.write(cm_df.to_string() + "\n\n")
            f.write(report_str + "\n")
    print(f"Saved {REPORTS_DIR / 'evaluation_detail.txt'}")


if __name__ == "__main__":
    main()
