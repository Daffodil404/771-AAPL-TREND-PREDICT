"""Earnings calendar from yfinance and row-level features for the price panel.

Yahoo's earnings calendar (what yfinance scrapes) is not a full IPO-to-present
history. Use ``get_earnings_dates(limit=100, offset=...)`` and paginate; even
then, AAPL typically reaches only back to the late 1990s. Earlier announcement
dates need another vendor or a hand-built list.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _normalize_earnings_index(raw: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(raw, errors="coerce")).dropna()
    if len(idx) == 0:
        return idx
    if idx.tz is not None:
        idx = idx.tz_convert("America/New_York")
        idx = pd.to_datetime(idx.strftime("%Y-%m-%d"))
    else:
        idx = idx.normalize()
    return pd.DatetimeIndex(np.unique(idx)).sort_values()


def fetch_earnings_dates(
    ticker: str,
    *,
    cache_path: Path | None = None,
    max_pages: int = 30,
) -> pd.DatetimeIndex:
    """Pull all earnings rows Yahoo exposes for this symbol (paginated).

    A fresh ``Ticker`` is used per request because yfinance caches
    ``get_earnings_dates(limit=...)`` by *limit only* and ignores *offset* on
    repeat calls on the same instance.
    """
    import yfinance as yf

    chunks: list[pd.DatetimeIndex] = []
    offset = 0
    for _ in range(max_pages):
        inst = yf.Ticker(ticker)
        ed = inst.get_earnings_dates(limit=100, offset=offset)
        if ed is None or getattr(ed, "empty", True):
            break
        chunks.append(pd.DatetimeIndex(ed.index))
        n = len(ed)
        offset += n
        if n < 100:
            break

    if not chunks:
        idx = pd.DatetimeIndex([])
    else:
        combined = np.unique(np.concatenate([c.values for c in chunks]))
        idx = _normalize_earnings_index(pd.DatetimeIndex(combined))

    if cache_path is not None and len(idx) > 0:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"earnings_date": idx}).to_csv(cache_path, index=False)
    return idx


def load_earnings_dates_cache(cache_path: Path) -> pd.DatetimeIndex:
    if not cache_path.exists():
        return pd.DatetimeIndex([])
    df = pd.read_csv(cache_path, parse_dates=["earnings_date"])
    return _normalize_earnings_index(pd.DatetimeIndex(df["earnings_date"]))


def load_or_fetch_earnings_dates(ticker: str, cache_path: Path) -> pd.DatetimeIndex:
    try:
        return fetch_earnings_dates(ticker, cache_path=cache_path)
    except Exception:
        cached = load_earnings_dates_cache(cache_path)
        if len(cached) > 0:
            return cached
        return pd.DatetimeIndex([])


def _normalize_panel_dates(series: pd.Series) -> np.ndarray:
    dates = pd.to_datetime(series, errors="coerce")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert("America/New_York")
        dates = pd.to_datetime(dates.dt.strftime("%Y-%m-%d"))
    else:
        dates = dates.dt.normalize()
    return dates.to_numpy(dtype="datetime64[D]")


def add_earnings_features(
    df: pd.DataFrame,
    earnings_idx: pd.DatetimeIndex,
    *,
    date_col: str = "date",
    unknown_sentinel: float = -1.0,
) -> pd.DataFrame:
    """Add is_earnings_date, days_since_earnings, days_until_earnings (trading calendar days)."""
    out = df.copy()
    d_np = _normalize_panel_dates(out[date_col])
    if len(earnings_idx) == 0:
        out["is_earnings_date"] = np.int8(0)
        out["days_since_earnings"] = unknown_sentinel
        out["days_until_earnings"] = unknown_sentinel
        return out

    e = _normalize_earnings_index(earnings_idx).to_numpy(dtype="datetime64[D]")
    in_set = np.isin(d_np, e)
    idx_prev = np.searchsorted(e, d_np, side="right") - 1
    days_since = np.where(
        idx_prev >= 0,
        (d_np - e[idx_prev]).astype("timedelta64[D]").astype(np.float64),
        unknown_sentinel,
    )
    idx_next = np.searchsorted(e, d_np, side="right")
    days_until = np.where(
        idx_next < len(e),
        (e[idx_next] - d_np).astype("timedelta64[D]").astype(np.float64),
        unknown_sentinel,
    )
    out["is_earnings_date"] = in_set.astype(np.int8)
    out["days_since_earnings"] = days_since
    out["days_until_earnings"] = days_until
    return out
