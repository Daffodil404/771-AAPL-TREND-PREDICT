from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
QUANTILE_DIR = REPORTS_DIR / "quantile"
THREED_DIR = REPORTS_DIR / "3-day-return"
HMM_DIR = PROJECT_ROOT / "results" / "experiments" / "hmm_triple_barrier"


def style():
    plt.style.use("seaborn-v0_8-whitegrid")


def save_fixed_vs_quantile():
    acc_fixed = pd.read_csv(REPORTS_DIR / "evaluation_accuracy.csv")
    acc_quant = pd.read_csv(QUANTILE_DIR / "evaluation_accuracy.csv")
    fixed_map = dict(zip(acc_fixed["method"], acc_fixed["accuracy"]))
    quant_map = dict(zip(acc_quant["method"], acc_quant["accuracy"]))

    methods = [
        "ridge",
        "logreg_ovr",
        "ensemble_lr_svm",
        "svm_linear",
        "rf_best_fast",
        "rf_best_slow",
        "lgbm",
    ]
    labels = [
        "Ridge",
        "LogReg OVR",
        "Ensemble",
        "Linear SVM",
        "RF Fast",
        "RF Slow",
        "LightGBM",
    ]
    df = pd.DataFrame(
        {
            "Model": labels,
            "Fixed": [fixed_map[m] for m in methods],
            "Quantile": [quant_map[m] for m in methods],
        }
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = range(len(df))
    width = 0.38
    ax.bar([i - width / 2 for i in x], df["Fixed"], width=width, label="Fixed epsilon", color="#355C7D")
    ax.bar([i + width / 2 for i in x], df["Quantile"], width=width, label="Quantile", color="#C06C84")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["Model"], rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Selected Models: Fixed Threshold vs Quantile Labeling")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "comparison_fixed_vs_quantile_selected_models.png", dpi=180)
    plt.close(fig)


def save_horizon_accuracy():
    df = pd.read_csv(THREED_DIR / "comparison_score_1day_vs_3day.csv")
    df = df.sort_values("acc_1day", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    x = range(len(df))
    width = 0.38
    ax.bar([i - width / 2 for i in x], df["acc_1day"], width=width, label="1-day", color="#2A9D8F")
    ax.bar([i + width / 2 for i in x], df["acc_3day"], width=width, label="3-day", color="#E76F51")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["method"], rotation=30, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Prediction Horizon Comparison")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "nextday_vs_3day_accuracy.png", dpi=180)
    plt.close(fig)


def save_3day_top_accuracy():
    df = pd.read_csv(THREED_DIR / "comparison_score_1day_vs_3day.csv")
    top = df.sort_values("acc_3day", ascending=False).head(6).sort_values("acc_3day", ascending=True)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(top["method"], top["acc_3day"], color="#F4A261")
    ax.set_xlabel("3-day Accuracy")
    ax.set_title("Top Models Under 3-day Horizon")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "3day_top_accuracy.png", dpi=180)
    plt.close(fig)


def save_hmm_experiment():
    df = pd.read_csv(HMM_DIR / "summary.csv")
    keep = ["ridge_base", "ridge_plus_regime_conf", "base", "base_plus_regime"]
    labels = {
        "ridge_base": "Ridge",
        "ridge_plus_regime_conf": "Ridge + Regime + Conf",
        "base": "LightGBM",
        "base_plus_regime": "LightGBM + Regime",
    }
    plot_df = df[df["variant"].isin(keep)].copy()
    plot_df["label"] = plot_df["variant"].map(labels)
    order = ["ridge_base", "ridge_plus_regime_conf", "base", "base_plus_regime"]
    plot_df["order"] = plot_df["variant"].map({name: i for i, name in enumerate(order)})
    plot_df = plot_df.sort_values("order")

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    colors = ["#6C757D", "#8D99AE", "#2A9D8F", "#264653"]
    ax.bar(plot_df["label"], plot_df["score_mean"], color=colors)
    ax.set_ylabel("Mean Composite Score")
    ax.set_title("HMM Regime + Triple Barrier Experiment")
    ax.set_ylim(0.28, 0.40)
    for idx, value in enumerate(plot_df["score_mean"]):
        ax.text(idx, value + 0.002, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "hmm_triple_barrier_comparison.png", dpi=180)
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    style()
    save_fixed_vs_quantile()
    save_horizon_accuracy()
    save_3day_top_accuracy()
    save_hmm_experiment()


if __name__ == "__main__":
    main()
