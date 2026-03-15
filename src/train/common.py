"""Shared paths, constants, and helpers for training scripts."""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repo root (src/train/ -> 2 up)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

TARGET_COLUMN = "target_label_next_day"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
# TEST_RATIO = 1 - TRAIN_RATIO - VAL_RATIO

FEATURE_COLUMNS = [
    "daily_return_rate",
    "overnight_return_rate",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "volume_5",
    "body_length",
    "return_3",
    "return_5",
    "rsi_14",
    "macd",
    "ma_ratio_5_21",
]


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe by time into train / val / test (no shuffle)."""
    n = len(df)
    i_train = int(n * TRAIN_RATIO)
    i_val = int(n * (TRAIN_RATIO + VAL_RATIO))
    train = df.iloc[:i_train]
    val = df.iloc[i_train:i_val]
    test = df.iloc[i_val:]
    return train, val, test


def load_features_and_split(
    feature_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load aapl_features.csv, parse date, dropna, sort, and return train/val/test."""
    if feature_path is None:
        feature_path = PROCESSED_DIR / "aapl_features.csv"
    df = pd.read_csv(feature_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna().sort_values("date").reset_index(drop=True)
    return time_split(df)
