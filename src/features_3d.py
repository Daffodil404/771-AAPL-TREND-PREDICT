"""Build features for 3-day trend prediction."""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # repo root (src/ -> 1 up)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FEATURE_COLUMNS = [
    "daily_return_rate",
    "overnight_return_rate",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "volatility_30",
    "volume_5",
    "body_length",
    "return_3",
    "return_5",
    "return_10",
    "rsi_14",
    "macd",
    "ma_ratio_5_21",
    "return_3_volatility_5",
    "nasdaq_return_1",
    "nasdaq_return_5",
    "relative_strength",
    "is_earnings_date",
    "days_since_earnings",
    "days_until_earnings",
]
TARGET_COLUMN = "target_label_next_3d"


def basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select feature columns + target, drop rows with NaN in any of them."""
    cols = ["date"] + FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in dataframe: {missing}")
    out = df[cols].copy()
    out = out.dropna()
    return out


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge NASDAQ (^IXIC) returns and add relative strength."""
    nasdaq_path = PROCESSED_DIR / "nasdaq_ixic.csv"
    nasdaq = pd.read_csv(nasdaq_path, index_col=0)
    nasdaq.index = pd.to_datetime(nasdaq.index, errors="coerce")
    nasdaq = nasdaq.dropna(subset=["Close"]).sort_index()

    if "return_1" not in nasdaq.columns:
        nasdaq["return_1"] = nasdaq["Close"].pct_change()
    if "return_5" not in nasdaq.columns:
        nasdaq["return_5"] = nasdaq["Close"].pct_change(5)

    nasdaq_feat = nasdaq[["return_1", "return_5"]].copy()
    nasdaq_feat = nasdaq_feat.rename(
        columns={"return_1": "nasdaq_return_1", "return_5": "nasdaq_return_5"}
    )
    nasdaq_feat.index.name = "date"
    nasdaq_feat = nasdaq_feat.reset_index()

    out = df.merge(nasdaq_feat, on="date", how="left")
    out["relative_strength"] = out["daily_return_rate"] - out["nasdaq_return_1"]
    return out


def main() -> None:
    input_path = PROCESSED_DIR / "aapl_processed.csv"
    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", TARGET_COLUMN])
    df = add_market_features(df)

    feature_df = basic_features(df)
    print(f"Feature matrix shape: {feature_df.shape}")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Target: {TARGET_COLUMN}")

    output_path = PROCESSED_DIR / "aapl_features_3d.csv"
    feature_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
