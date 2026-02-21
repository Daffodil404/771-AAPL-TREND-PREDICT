"""Build features for trend prediction."""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # repo root (src/ -> 1 up)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# 当前使用的特征列（可按验证集效果在 features.py 里加/减）
FEATURE_COLUMNS = [
    "daily_return_rate",
    "overnight_return_rate",
    "volatility_5",
    "volatility_10",
    "volume_5",
    "body_length",
]
TARGET_COLUMN = "target_label_next_day"


def basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select feature columns + target, drop rows with NaN in any of them."""
    cols = ["date"] + FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in dataframe: {missing}")
    out = df[cols].copy()
    out = out.dropna()
    return out


def main() -> None:
    input_path = PROCESSED_DIR / "aapl_processed.csv"
    df = pd.read_csv(input_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", TARGET_COLUMN])

    feature_df = basic_features(df)
    print(f"Feature matrix shape: {feature_df.shape}")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Target: {TARGET_COLUMN}")

    output_path = PROCESSED_DIR / "aapl_features.csv"
    feature_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
