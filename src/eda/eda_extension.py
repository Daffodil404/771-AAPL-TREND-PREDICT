from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"


def infer_label_from_return(ret: pd.Series, flat_threshold: float) -> pd.Series:
    out = np.where(ret > flat_threshold, "up", np.where(ret < -flat_threshold, "down", "flat"))
    return pd.Series(out, index=ret.index)


def plot_threshold_sensitivity(df: pd.DataFrame, thresholds: list[float]) -> tuple[pd.DataFrame, list[Path]]:
    if "daily_return_rate" not in df.columns:
        raise ValueError("Column 'daily_return_rate' not found in processed data.")

    rows = []
    for th in thresholds:
        labels = infer_label_from_return(df["daily_return_rate"], th)
        counts = labels.value_counts().reindex(["up", "down", "flat"], fill_value=0)
        total = counts.sum()
        majority_acc = counts.max() / total if total else np.nan
        rows.append(
            {
                "flat_threshold": th,
                "up_pct": counts["up"] / total * 100,
                "down_pct": counts["down"] / total * 100,
                "flat_pct": counts["flat"] / total * 100,
                "majority_baseline_acc": majority_acc,
                "majority_label": counts.idxmax(),
            }
        )

    summary = pd.DataFrame(rows).sort_values("flat_threshold")
    print("\n[Threshold Sensitivity Summary]")
    print(summary.round(4).to_string(index=False))

    saved: list[Path] = []

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(summary))
    up = summary["up_pct"].to_numpy()
    down = summary["down_pct"].to_numpy()
    flat = summary["flat_pct"].to_numpy()
    ax.bar(x, up, label="up")
    ax.bar(x, down, bottom=up, label="down")
    ax.bar(x, flat, bottom=up + down, label="flat")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in summary["flat_threshold"]])
    ax.set_ylabel("Percentage")
    ax.set_xlabel("Flat threshold")
    ax.set_title("Class composition vs flat threshold")
    ax.legend()
    plt.tight_layout()
    p1 = FIGURE_DIR / "eda_threshold_class_composition.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    saved.append(p1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(summary["flat_threshold"], summary["majority_baseline_acc"], marker="o")
    ax.set_xlabel("Flat threshold")
    ax.set_ylabel("Majority baseline accuracy")
    ax.set_title("Majority baseline accuracy vs flat threshold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p2 = FIGURE_DIR / "eda_threshold_baseline_accuracy.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    saved.append(p2)

    for p in saved:
        print(f"Saved plot: {p}")
    return summary, saved


def plot_feature_distribution_by_label(df: pd.DataFrame, label_col: str = "target_label_next_day") -> list[Path]:
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in processed data.")

    candidate_features = [
        "daily_return_rate",
        "ma5",
        "ma10",
        "ma20",
        "ma_diff",
        "volume_change",
        "volatility_5d",
        "volatility_10d",
    ]
    features = [c for c in candidate_features if c in df.columns]

    if not features:
        print("No candidate feature columns found for feature-by-label EDA. Skipping.")
        return []

    saved: list[Path] = []
    for feat in features:
        sub = df[[feat, label_col]].dropna().copy()
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 4))
        order = [x for x in ["down", "flat", "up"] if x in sub[label_col].unique()]
        data = [sub.loc[sub[label_col] == c, feat].to_numpy() for c in order]
        ax.boxplot(data, tick_labels=order, showfliers=False)
        ax.set_title(f"{feat} distribution by label")
        ax.set_ylabel(feat)
        ax.set_xlabel("label")
        plt.tight_layout()
        out1 = FIGURE_DIR / f"eda_box_{feat}_by_label.png"
        fig.savefig(out1, dpi=150)
        plt.close(fig)
        saved.append(out1)

        fig, ax = plt.subplots(figsize=(8, 4))
        for c in order:
            vals = sub.loc[sub[label_col] == c, feat].dropna().to_numpy()
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=50, alpha=0.35, density=True, label=c)
        ax.set_title(f"{feat} histogram by label")
        ax.set_xlabel(feat)
        ax.set_ylabel("density")
        ax.legend()
        plt.tight_layout()
        out2 = FIGURE_DIR / f"eda_hist_{feat}_by_label.png"
        fig.savefig(out2, dpi=150)
        plt.close(fig)
        saved.append(out2)

    for p in saved:
        print(f"Saved plot: {p}")
    return saved


def plot_time_split_overview(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15) -> Path | None:
    if "date" not in df.columns:
        raise ValueError("Column 'date' not found in processed data.")

    ts = df.sort_values("date").reset_index(drop=True).copy()
    n = len(ts)
    if n == 0:
        print("No rows to split. Skipping time split overview.")
        return None

    i_train = int(n * train_ratio)
    i_val = int(n * (train_ratio + val_ratio))
    ts["idx"] = np.arange(n)

    fig, ax = plt.subplots(figsize=(12, 4))
    y_col = "close" if "close" in ts.columns else ("adj_close" if "adj_close" in ts.columns else None)
    if y_col is not None:
        ax.plot(ts["idx"], ts[y_col], color="#1f77b4", linewidth=1.2, label=y_col)

    ax.axvspan(0, i_train, alpha=0.10, color="green", label="train")
    ax.axvspan(i_train, i_val, alpha=0.10, color="orange", label="val")
    ax.axvspan(i_val, n - 1, alpha=0.10, color="red", label="test")

    def safe_date(i: int) -> str:
        i = min(max(i, 0), n - 1)
        return str(ts.loc[i, "date"].date())

    ax.set_title(
        "Time-based train/val/test split\n"
        f"train: {safe_date(0)} ~ {safe_date(i_train-1)}, "
        f"val: {safe_date(i_train)} ~ {safe_date(i_val-1)}, "
        f"test: {safe_date(i_val)} ~ {safe_date(n-1)}"
    )
    ax.set_xlabel("time index")
    ax.set_ylabel(y_col or "")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)
    plt.tight_layout()

    out = FIGURE_DIR / "eda_time_split_overview.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out}")
    return out


def build_html_report(
    summary: pd.DataFrame,
    threshold_plots: list[Path],
    feature_plots: list[Path],
    split_plot: Path | None,
    out_html: Path,
) -> None:
    def img_tag(p: Path) -> str:
        rel = p.relative_to(out_html.parent)
        return f'<div style="margin:16px 0"><img src="{rel.as_posix()}" style="max-width:100%;border:1px solid #ddd;border-radius:8px"/></div>'

    html = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>EDA Report</title></head>",
        "<body style='font-family:Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 12px'>",
        "<h1>AAPL Trend EDA Report (Auto-generated)</h1>",
        "<h2>1) Flat Threshold Sensitivity</h2>",
        summary.round(4).to_html(index=False),
    ]
    html += [img_tag(p) for p in threshold_plots]

    html.append("<h2>2) Feature Distribution by Label</h2>")
    if feature_plots:
        html += [img_tag(p) for p in feature_plots]
    else:
        html.append("<p>No feature plots generated.</p>")

    html.append("<h2>3) Time Split Overview</h2>")
    if split_plot:
        html.append(img_tag(split_plot))
    else:
        html.append("<p>No split plot generated.</p>")

    html.append("</body></html>")
    out_html.write_text("\n".join(html), encoding="utf-8")
    print(f"Saved report: {out_html}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extended EDA for 771_project")
    parser.add_argument("--input", type=str, default=str(PROCESSED_DIR / "aapl_processed.csv"))
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.002, 0.005, 0.01], help="flat threshold in decimal, e.g. 0.002 = 0.2%%")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--report_name", type=str, default="eda_extension_report.html")
    args = parser.parse_args()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).copy()

    summary, threshold_plots = plot_threshold_sensitivity(df, thresholds=args.thresholds)
    summary_csv = PROCESSED_DIR / "eda_threshold_sensitivity_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"Saved table: {summary_csv}")

    feature_plots = plot_feature_distribution_by_label(df, label_col="target_label_next_day")
    split_plot = plot_time_split_overview(df, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    report_path = FIGURE_DIR / args.report_name
    build_html_report(summary, threshold_plots, feature_plots, split_plot, report_path)
    print("\nExtended EDA finished.")


if __name__ == "__main__":
    main()
