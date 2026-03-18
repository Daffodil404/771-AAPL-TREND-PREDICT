"""Shared paths, constants, and helpers for quantile-label training scripts."""

from pathlib import Path
from typing import Optional
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # repo root (src/train/quantile/ -> 3 up)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "quantile"
RESULTS_DIR = PROJECT_ROOT / "results" / "quantile"

TARGET_COLUMN = "target_label_next_day_quantile"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

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
]

TREE_FEATURE_COLUMNS = [
    "return_3",
    "return_10",
    "volatility_10",
    "volatility_30",
]


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    i_train = int(n * TRAIN_RATIO)
    i_val = int(n * (TRAIN_RATIO + VAL_RATIO))
    train = df.iloc[:i_train]
    val = df.iloc[i_train:i_val]
    test = df.iloc[i_val:]
    return train, val, test


def load_features_and_split(
    feature_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if feature_path is None:
        feature_path = PROCESSED_DIR / "aapl_features.csv"
    df = pd.read_csv(feature_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna().sort_values("date").reset_index(drop=True)
    return time_split(df)
