"""Train and evaluate baseline methods: random, momentum, moving-average."""

import numpy as np
import pandas as pd

from common import (
    PROCESSED_DIR,
    RESULTS_DIR,
    TARGET_COLUMN,
    load_features_and_split,
)

# 与 base.py 一致，用于 momentum 的 up/down/flat 分界
EPS = 0.005


def train_random_baseline(test_df: pd.DataFrame, train_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Random baseline：按训练集类别比例随机抽样预测 test。"""
    rng = np.random.default_rng(seed)
    probs = train_df[TARGET_COLUMN].value_counts(normalize=True).reindex(["up", "down", "flat"], fill_value=0)
    probs = probs / probs.sum()
    pred = rng.choice(probs.index, size=len(test_df), p=probs.values)
    out = test_df[["date"]].copy()
    out["true_label"] = test_df[TARGET_COLUMN].values
    out["pred_label"] = pred
    return out


def train_momentum_baseline(test_df: pd.DataFrame) -> pd.DataFrame:
    """Momentum baseline：用当日 daily_return_rate 的符号/阈值预测下一日方向。"""
    r = test_df["daily_return_rate"]
    pred = np.where(r > EPS, "up", np.where(r < -EPS, "down", "flat"))
    out = test_df[["date"]].copy()
    out["true_label"] = test_df[TARGET_COLUMN].values
    out["pred_label"] = pred
    return out


def train_moving_average_baseline(
    test_df: pd.DataFrame, processed_df: pd.DataFrame
) -> pd.DataFrame:
    """Moving-average baseline：用 close 的 MA5 vs MA20 判断 up/down/flat。"""
    close = processed_df.set_index("date")["close"].sort_index()
    ma5 = close.rolling(5, min_periods=1).mean()
    ma20 = close.rolling(20, min_periods=1).mean()
    # 与 test 的 date 对齐
    test_dates = test_df["date"].values
    ma5_test = ma5.reindex(test_dates).values
    ma20_test = ma20.reindex(test_dates).values
    pred = np.where(ma5_test > ma20_test, "up", np.where(ma5_test < ma20_test, "down", "flat"))
    out = test_df[["date"]].copy()
    out["true_label"] = test_df[TARGET_COLUMN].values
    out["pred_label"] = pred
    return out


def main() -> None:
    train_df, val_df, test_df = load_features_and_split()
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 2) Random baseline
    pred_random = train_random_baseline(test_df, train_df)
    pred_random.to_csv(RESULTS_DIR / "pred_random.csv", index=False)
    print(f"Saved {RESULTS_DIR / 'pred_random.csv'}")

    # 3) Momentum baseline
    pred_momentum = train_momentum_baseline(test_df)
    pred_momentum.to_csv(RESULTS_DIR / "pred_momentum.csv", index=False)
    print(f"Saved {RESULTS_DIR / 'pred_momentum.csv'}")

    # 4) MA baseline：需要 close，从 processed 读
    processed_path = PROCESSED_DIR / "aapl_processed.csv"
    processed = pd.read_csv(processed_path)
    processed["date"] = pd.to_datetime(processed["date"], errors="coerce")
    pred_ma = train_moving_average_baseline(test_df, processed)
    pred_ma.to_csv(RESULTS_DIR / "pred_ma.csv", index=False)
    print(f"Saved {RESULTS_DIR / 'pred_ma.csv'}")

    print("Baseline training done.")


if __name__ == "__main__":
    main()
