"""HMM regime + Triple Barrier LightGBM experiment.

This experiment is intentionally isolated from the fixed-threshold / quantile
pipelines. It builds a leakage-safe regime feature by fitting an HMM on a
rolling lookback window, then compares LightGBM with and without the regime ID
using TimeSeriesSplit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import contextlib
import io
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, recall_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "hmmlearn is required for this experiment. Install it with "
        "`python3 -m pip install hmmlearn`."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "experiments" / "hmm_triple_barrier"

LABELS = ["down", "flat", "up"]
STATE_NAMES = {
    0: "high_vol_down",
    1: "low_vol_sideways",
    2: "medium_vol_up",
}


@dataclass
class ExperimentConfig:
    max_rows: int = 2500
    regime_window: int = 60
    n_states: int = 3
    n_splits: int = 5
    vertical_horizon: int = 5
    barrier_mult: float = 1.5
    random_state: int = 42


def load_processed() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "aapl_processed.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    nasdaq = pd.read_csv(PROCESSED_DIR / "nasdaq_ixic.csv", index_col=0)
    nasdaq.index = pd.to_datetime(nasdaq.index, errors="coerce")
    nasdaq = nasdaq.dropna(subset=["Close"]).sort_index()
    if "return_1" not in nasdaq.columns:
        nasdaq["return_1"] = nasdaq["Close"].pct_change()
    if "return_5" not in nasdaq.columns:
        nasdaq["return_5"] = nasdaq["Close"].pct_change(5)
    feat = (
        nasdaq[["return_1", "return_5"]]
        .rename(columns={"return_1": "nasdaq_return_1", "return_5": "nasdaq_return_5"})
        .rename_axis("date")
        .reset_index()
    )
    out = df.merge(feat, on="date", how="left")
    out["relative_strength"] = out["daily_return_rate"] - out["nasdaq_return_1"]
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_market_features(df.copy())
    out["log_return"] = np.log(out["close"]).diff()
    out["open_log_return"] = np.log(out["open"]).diff()
    out["high_log_return"] = np.log(out["high"]).diff()
    out["low_log_return"] = np.log(out["low"]).diff()
    out["close_log_return"] = np.log(out["close"]).diff()
    out["log_return_3"] = np.log(out["close"] / out["close"].shift(3))
    out["log_return_5"] = np.log(out["close"] / out["close"].shift(5))
    out["log_volume_return"] = np.log1p(out["volume"]).diff()
    out["hl_log_range"] = np.log(out["high"] / out["low"])
    out["oc_log_return"] = np.log(out["close"] / out["open"])
    out["realized_vol_20"] = out["log_return"].rolling(20).std()
    out["realized_vol_60"] = out["log_return"].rolling(60).std()
    return out


def triple_barrier_labels(
    close: pd.Series,
    vol: pd.Series,
    *,
    horizon: int,
    barrier_mult: float,
) -> pd.Series:
    labels = pd.Series(pd.NA, index=close.index, dtype="object")
    for i in range(len(close) - horizon):
        price0 = close.iloc[i]
        sigma = vol.iloc[i]
        if pd.isna(price0) or pd.isna(sigma) or sigma <= 0:
            continue
        upper = barrier_mult * sigma
        lower = -barrier_mult * sigma
        future = close.iloc[i + 1 : i + horizon + 1]
        path_ret = np.log(future / price0)
        label = "flat"
        for ret in path_ret:
            if ret >= upper:
                label = "up"
                break
            if ret <= lower:
                label = "down"
                break
        labels.iloc[i] = label
    return labels


def _map_states_to_semantics(states: np.ndarray, window_df: pd.DataFrame, n_states: int) -> dict[int, int]:
    stats = []
    for state in range(n_states):
        mask = states == state
        if not mask.any():
            stats.append((state, np.inf, -np.inf))
            continue
        mean_ret = float(window_df.loc[mask, "close_log_return"].mean())
        mean_vol = float(window_df.loc[mask, "close_log_return"].std())
        stats.append((state, mean_vol, mean_ret))

    down_state = sorted(stats, key=lambda x: (x[2], -x[1]))[0][0]
    up_state = sorted(stats, key=lambda x: (-x[2], x[1]))[0][0]
    remaining = [s[0] for s in stats if s[0] not in {down_state, up_state}]
    sideways_state = remaining[0] if remaining else up_state

    return {
        down_state: 0,
        sideways_state: 1,
        up_state: 2,
    }


def rolling_hmm_regimes(
    df: pd.DataFrame,
    *,
    window: int,
    n_states: int,
    random_state: int,
) -> pd.DataFrame:
    hmm_features = [
        "open_log_return",
        "high_log_return",
        "low_log_return",
        "close_log_return",
        "log_volume_return",
    ]
    regime_ids = np.full(len(df), np.nan)
    regime_conf = np.full(len(df), np.nan)

    for end in range(window - 1, len(df)):
        window_df = df.iloc[end - window + 1 : end + 1][hmm_features].dropna()
        if len(window_df) < max(30, n_states * 10):
            continue
        X = StandardScaler().fit_transform(window_df[hmm_features].to_numpy())
        try:
            model = GaussianHMM(
                n_components=n_states,
                covariance_type="diag",
                n_iter=100,
                random_state=random_state,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with contextlib.redirect_stderr(io.StringIO()):
                    model.fit(X)
                    states = model.predict(X)
                    current_proba = model.predict_proba(X)[-1]
            state_map = _map_states_to_semantics(states, window_df, n_states)
            current_state = int(states[-1])
        except Exception:
            continue

        regime_ids[end] = state_map.get(current_state, np.nan)
        regime_conf[end] = float(current_proba[current_state])

    out = df.copy()
    out["regime_id"] = regime_ids
    out["regime_confidence"] = regime_conf
    out["regime_name"] = out["regime_id"].map(STATE_NAMES)
    return out


def prepare_dataset(config: ExperimentConfig) -> pd.DataFrame:
    df = engineer_features(load_processed())
    if config.max_rows:
        df = df.tail(config.max_rows).reset_index(drop=True)
    df["tb_vol"] = df["log_return"].ewm(span=20, min_periods=20).std()
    df["target_tb_next_day"] = triple_barrier_labels(
        df["close"],
        df["tb_vol"],
        horizon=config.vertical_horizon,
        barrier_mult=config.barrier_mult,
    )
    df = rolling_hmm_regimes(
        df,
        window=config.regime_window,
        n_states=config.n_states,
        random_state=config.random_state,
    )
    return df


def experiment_features(include_regime: bool) -> list[str]:
    cols = [
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
        "log_return",
        "log_return_3",
        "log_return_5",
        "log_volume_return",
        "hl_log_range",
        "oc_log_return",
        "realized_vol_20",
        "realized_vol_60",
    ]
    if include_regime:
        cols += ["regime_id", "regime_confidence"]
    return cols


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    macro_recall = recall_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    min_recall = float(np.min(per_class_recall)) if len(per_class_recall) else 0.0
    score = 0.5 * acc + 0.3 * macro_recall + 0.2 * min_recall
    return {
        "accuracy": acc,
        "macro_recall": macro_recall,
        "min_recall": min_recall,
        "score": score,
    }


def run_tscv_lightgbm(df: pd.DataFrame, *, include_regime: bool, config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = experiment_features(include_regime)
    cols = ["date", "target_tb_next_day"] + features
    work = df[cols].dropna().copy()
    work["target_tb_next_day"] = pd.Categorical(work["target_tb_next_day"], categories=LABELS)
    if include_regime:
        work["regime_id"] = work["regime_id"].astype(int).astype("category")

    X = work[features]
    y = work["target_tb_next_day"]
    tscv = TimeSeriesSplit(n_splits=config.n_splits)
    fold_rows = []
    pred_parts = []

    for fold_id, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        train_end = int(len(train_idx) * 0.85)
        inner_train_idx = train_idx[:train_end]
        val_idx = train_idx[train_end:]
        if len(val_idx) == 0:
            continue

        X_train = X.iloc[inner_train_idx].copy()
        y_train = y.iloc[inner_train_idx].copy()
        X_val = X.iloc[val_idx].copy()
        y_val = y.iloc[val_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_test = y.iloc[test_idx].copy()

        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=31,
            min_child_samples=20,
            reg_alpha=0.0,
            reg_lambda=1.0,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=config.random_state,
            verbosity=-1,
        )
        fit_kwargs = {
            "eval_set": [(X_val, y_val)],
            "eval_metric": "multi_logloss",
            "callbacks": [lgb.early_stopping(stopping_rounds=50, verbose=False)],
        }
        if include_regime:
            fit_kwargs["categorical_feature"] = ["regime_id"]
        model.fit(X_train, y_train, **fit_kwargs)
        pred = pd.Series(model.predict(X_test, num_iteration=model.best_iteration_), index=y_test.index)
        metrics = evaluate_predictions(y_test, pred)
        fold_rows.append(
            {
                "variant": "base_plus_regime" if include_regime else "base",
                "fold": fold_id,
                "train_n": len(X_train),
                "val_n": len(X_val),
                "test_n": len(X_test),
                "best_iteration": int(model.best_iteration_ or 0),
                **metrics,
            }
        )
        pred_parts.append(
            pd.DataFrame(
                {
                    "date": work.loc[y_test.index, "date"].values,
                    "variant": "base_plus_regime" if include_regime else "base",
                    "fold": fold_id,
                    "true_label": y_test.astype(str).values,
                    "pred_label": pred.astype(str).values,
                }
            )
        )

    return pd.DataFrame(fold_rows), pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()


def save_summary(base_rows: pd.DataFrame, regime_rows: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([base_rows, regime_rows], ignore_index=True)
    summary = (
        combined.groupby("variant")[["accuracy", "macro_recall", "min_recall", "score"]]
        .agg(["mean", "std"])
    )
    summary.columns = ["_".join(col) for col in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
    return summary


def main() -> None:
    config = ExperimentConfig()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = prepare_dataset(config)
    df.to_csv(RESULTS_DIR / "dataset_with_regimes.csv", index=False)

    base_rows, base_preds = run_tscv_lightgbm(df, include_regime=False, config=config)
    regime_rows, regime_preds = run_tscv_lightgbm(df, include_regime=True, config=config)

    base_rows.to_csv(RESULTS_DIR / "fold_metrics_base.csv", index=False)
    regime_rows.to_csv(RESULTS_DIR / "fold_metrics_base_plus_regime.csv", index=False)
    if not base_preds.empty:
        base_preds.to_csv(RESULTS_DIR / "predictions_base.csv", index=False)
    if not regime_preds.empty:
        regime_preds.to_csv(RESULTS_DIR / "predictions_base_plus_regime.csv", index=False)

    summary = save_summary(base_rows, regime_rows)
    with open(RESULTS_DIR / "report.txt", "w", encoding="utf-8") as f:
        f.write("HMM Regime + Triple Barrier Experiment\n")
        f.write("=" * 40 + "\n\n")
        f.write("Configuration\n")
        f.write(f"- max_rows: {config.max_rows}\n")
        f.write(f"- regime_window: {config.regime_window}\n")
        f.write(f"- n_states: {config.n_states}\n")
        f.write(f"- TimeSeriesSplit folds: {config.n_splits}\n")
        f.write(f"- vertical_horizon: {config.vertical_horizon}\n")
        f.write(f"- barrier_mult: {config.barrier_mult}\n\n")
        f.write("Summary\n")
        f.write(summary.to_string(index=False))
        f.write("\n\n")
        for name, rows, preds in [
            ("base", base_rows, base_preds),
            ("base_plus_regime", regime_rows, regime_preds),
        ]:
            f.write(f"[{name}]\n")
            if rows.empty or preds.empty:
                f.write("No results.\n\n")
                continue
            overall = evaluate_predictions(preds["true_label"], preds["pred_label"])
            f.write(f"overall_accuracy={overall['accuracy']:.4f}\n")
            f.write(f"overall_macro_recall={overall['macro_recall']:.4f}\n")
            f.write(f"overall_min_recall={overall['min_recall']:.4f}\n")
            f.write(f"overall_score={overall['score']:.4f}\n")
            f.write(classification_report(preds["true_label"], preds["pred_label"], labels=LABELS, zero_division=0))
            f.write("\n\n")

    print(f"Saved experiment outputs to {RESULTS_DIR}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
