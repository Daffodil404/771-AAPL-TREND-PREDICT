from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repo root (src/ -> 2 up)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

def check_overall_label_distribution(df: pd.DataFrame) -> pd.Series:
    counts = df["target_label_next_day"].value_counts(dropna=False)
    pct = (counts / counts.sum() * 100).round(2)
    summary = pd.DataFrame({"count": counts, "pct": pct})
    print("\nOverall target_label_next_day distribution:")
    print(summary.to_string())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title("Target label (next day) distribution")
    plt.tight_layout()
    output_path = FIGURE_DIR / "overall_target_label_distribution.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {output_path}")
    return counts

def check_yearly_label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out["date"].dt.year

    yearly_count = (
        out.groupby(["year", "target_label_next_day"])
        .size()
        .unstack(fill_value=0)
    )
    yearly_ratio = yearly_count.div(yearly_count.sum(axis=1), axis=0)

    print("\nYearly target_label_next_day ratio:")
    print((yearly_ratio * 100).round(2).to_string())

    fig, ax = plt.subplots(figsize=(12, 5))
    yearly_ratio.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Yearly target label composition")
    ax.set_xlabel("Year")
    ax.set_ylabel("Ratio")
    ax.legend(title="label", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    output_path = FIGURE_DIR / "yearly_target_label_composition.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {output_path}")
    return yearly_ratio

# Compute baseline accuracy
# Result: Majority label: up, baseline accuracy: 46.79%
def compute_baseline(df: pd.DataFrame) -> pd.DataFrame:
    counts = df['target_label_next_day'].value_counts()
    majority_label = counts.idxmax()
    baseline_acc = counts[majority_label] / counts.sum()
    print(f"Baseline accuracy: {baseline_acc:.4f}")
    print(f"Majority label: {majority_label}")
    return baseline_acc

# Compute transition matrix (prev day → next day)’
# Transition Matrix (prev day → next day):
# target_label_next_day   down   flat     up
# prev_label                                
# down                   44.73   8.62  46.65
# flat                   44.61  10.99  44.40
# up                     43.25   9.34  47.40
# Nearly independent of the prev day.
def compute_transition_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df_sorted = df.sort_values(by="date").copy()
    df_sorted['prev_label'] = df_sorted['target_label_next_day'].shift(1)

    matrix = pd.crosstab(df_sorted['prev_label'], df_sorted['target_label_next_day'], normalize='index')
    print("\nTransition Matrix (prev day → next day):")
    print((matrix * 100).round(2).to_string())
    return matrix

def main() -> None:
    input_path = PROCESSED_DIR / "aapl_processed.csv"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "target_label_next_day"])

    check_overall_label_distribution(df)
    check_yearly_label_distribution(df)
    compute_baseline(df)
    compute_transition_matrix(df)


if __name__ == "__main__":
    main()
