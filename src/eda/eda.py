from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter


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
# Result: Majority label: up, baseline accuracy: 46.91%
def compute_baseline(df: pd.DataFrame) -> float:
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
# down                   44.63   8.63  46.73
# flat                   44.90  10.73  44.37
# up                     43.13   9.30  47.57
# Nearly independent of the prev day.
def compute_transition_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df_sorted = df.sort_values(by="date").copy()
    df_sorted['prev_label'] = df_sorted['target_label_next_day'].shift(1)

    matrix = pd.crosstab(df_sorted['prev_label'], df_sorted['target_label_next_day'], normalize='index')
    print("\nTransition Matrix (prev day → next day):")
    print((matrix * 100).round(2).to_string())
    return matrix


def show_close_price_trend(df: pd.DataFrame) -> None:
    # Show only the data
    formatter = DateFormatter('%Y-%m-%d')

    plt.figure(figsize = (18, 9))
    plt.plot(range(df.shape[0]), df['close'])
    plt.xticks(range(0, df.shape[0], 500), df['date'].loc[::500], rotation=45)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close price', fontsize=18)
    plt.title('Close price of Apple\'s shares')
    plt.gcf().axes[0].xaxis.set_major_formatter(formatter)
    plt.show()

def show_volume_trend(df: pd.DataFrame) -> None:
    """Daily volume time series."""
    formatter = DateFormatter("%Y-%m-%d")
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.plot(range(df.shape[0]), df["volume"])
    ax.set_xticks(range(0, df.shape[0], 500))
    ax.set_xticklabels(df["date"].iloc[::500], rotation=45)
    ax.set_xlabel("Date", fontsize=18)
    ax.set_ylabel("Volume", fontsize=18)
    ax.set_title("Volume of Apple's shares")
    ax.xaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.show()


def show_volume_by_year(df: pd.DataFrame) -> None:
    """Volume aggregated by year (total per year), bar chart."""
    agg = df.assign(year=df["date"].dt.year).groupby("year")["volume"].sum()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(agg.index.astype(str), agg.values, color="steelblue", edgecolor="navy", alpha=0.8)
    ax.set_xlabel("Year", fontsize=18)
    ax.set_ylabel("Total volume", fontsize=18)
    ax.set_title("Volume of Apple's shares (by year)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def show_volume_by_month(df: pd.DataFrame) -> None:
    """Volume aggregated by month (total per month), line chart."""
    agg = df.assign(ym=df["date"].dt.to_period("M")).groupby("ym")["volume"].sum()
    agg.index = agg.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(agg.index, agg.values)
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    ax.set_xlabel("Date", fontsize=18)
    ax.set_ylabel("Total volume", fontsize=18)
    ax.set_title("Volume of Apple's shares (by month)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main() -> None:
    input_path = PROCESSED_DIR / "aapl_processed.csv"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "target_label_next_day"])

    show_close_price_trend(df)
    show_volume_by_year(df)
    show_volume_by_month(df)
    check_overall_label_distribution(df)
    check_yearly_label_distribution(df)
    compute_baseline(df)
    compute_transition_matrix(df)


if __name__ == "__main__":
    main()
