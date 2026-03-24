"""Evaluate quantile-labeled prediction results."""

from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = PROJECT_ROOT / "results" / "quantile"
REPORTS_DIR = PROJECT_ROOT / "reports" / "quantile"

PRED_FILES = {
    "random": "pred_random.csv",
    "momentum": "pred_momentum.csv",
    "ma": "pred_ma.csv",
    "logreg_multinomial": "pred_logreg_multinomial.csv",
    "logreg_ovr": "pred_logreg_ovr.csv",
    "svm_linear": "pred_svm_linear.csv",
    "ensemble_lr_svm": "pred_ensemble.csv",
    "rf5": "pred_rf5.csv",
    "rf6": "pred_rf6.csv",
    "rf_best_fast": "pred_rf_best_fast.csv",
    "rf_best_slow": "pred_rf_best_slow.csv",
    "gb": "pred_gb.csv",
    "hgb_best_fast": "pred_hgb_best_fast.csv",
    "lgbm": "pred_lgbm.csv",
}

# Optional filter: comma-separated method names
import os as _os
_methods_env = _os.getenv("PIPELINE_METHODS", "").strip()
if _methods_env:
    _allowed = {m.strip() for m in _methods_env.split(",") if m.strip()}
    PRED_FILES = {k: v for k, v in PRED_FILES.items() if k in _allowed}


def evaluate_one(pred_path: Path) -> tuple[float, float, float, pd.DataFrame, str]:
    df = pd.read_csv(pred_path)
    y_true = df["true_label"]
    y_pred = df["pred_label"]
    acc = accuracy_score(y_true, y_pred)
    macro_recall = recall_score(y_true, y_pred, labels=["down", "flat", "up"], average="macro", zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, labels=["down", "flat", "up"], average=None, zero_division=0)
    min_recall = float(per_class_recall.min()) if len(per_class_recall) else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=["down", "flat", "up"])
    cm_df = pd.DataFrame(
        cm,
        index=["true_down", "true_flat", "true_up"],
        columns=["pred_down", "pred_flat", "pred_up"],
    )
    report_str = classification_report(y_true, y_pred, labels=["down", "flat", "up"], zero_division=0)
    return acc, macro_recall, min_recall, cm_df, report_str


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    no_reports = _os.getenv("PIPELINE_NO_REPORTS", "").strip() in {"1", "true", "yes", "y"}
    rows = []
    score_rows = []
    all_reports = []

    for name, filename in PRED_FILES.items():
        path = RESULTS_DIR / filename
        if not path.exists():
            print(f"Skip (not found): {path}")
            continue
        acc, macro_recall, min_recall, cm_df, report_str = evaluate_one(path)
        score = 0.5 * acc + 0.3 * macro_recall + 0.2 * min_recall
        rows.append({"method": name, "accuracy": acc})
        score_rows.append(
            {
                "method": name,
                "accuracy": acc,
                "macro_recall": macro_recall,
                "min_recall": min_recall,
                "score": score,
            }
        )
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"Macro Recall: {macro_recall:.4f}")
        print(f"Min Recall: {min_recall:.4f}")
        print(f"Score: {score:.4f}")
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
        if not no_reports:
            acc_df.to_csv(REPORTS_DIR / "evaluation_accuracy.csv", index=False)
            print(f"\nSaved {REPORTS_DIR / 'evaluation_accuracy.csv'}")

    score_df = pd.DataFrame(score_rows)
    if not score_df.empty:
        score_df = score_df.sort_values("score", ascending=False)
        print("\n" + "=" * 50)
        print("Score comparison")
        print(score_df.to_string(index=False))
        if not no_reports:
            score_df.to_csv(REPORTS_DIR / "evaluation_score.csv", index=False)
            print(f"\nSaved {REPORTS_DIR / 'evaluation_score.csv'}")

    if not no_reports:
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
