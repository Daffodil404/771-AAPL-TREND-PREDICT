import pandas as pd
import numpy as np

EPS = 0.005


def classify_three_way(rate: pd.Series) -> pd.Series:
    labels = np.where(rate > EPS, 'up', np.where(rate < -EPS, 'down', 'flat'))
    out = pd.Series(labels, index=rate.index, dtype="object")
    out[rate.isna()] = pd.NA
    return out

def calculate_daily_range(df: pd.DataFrame) -> pd.DataFrame:
    return (df['high'] - df['low']) / df['close']

def calculate_intraday_returns(df: pd.DataFrame) -> pd.DataFrame:
    return df['close'] - df['open']

def calculate_intraday_return_rate(df: pd.DataFrame) -> pd.DataFrame:
    return df['intraday_return'] / df['open']

def calculate_intraday_labels(df: pd.DataFrame) -> pd.DataFrame:
    return classify_three_way(df['intraday_return_rate'])

def calculate_daily_return(df: pd.DataFrame) -> pd.DataFrame:
    return df['close'] - df['close'].shift(1)

def calculate_daily_return_rate(df: pd.DataFrame) -> pd.DataFrame:
    return df['daily_return'] / df['close'].shift(1)

def calculate_daily_labels(df: pd.DataFrame) -> pd.DataFrame:
    return classify_three_way(df['daily_return_rate'])

def calculate_overnight_return(df: pd.DataFrame) -> pd.DataFrame:
    return df['open'] - df['close'].shift(1)

def calculate_overnight_return_rate(df: pd.DataFrame) -> pd.DataFrame:
    return df['overnight_return'] / df['close'].shift(1)

def calculate_overnight_labels(df: pd.DataFrame) -> pd.DataFrame:
    return classify_three_way(df['overnight_return_rate'])


def calculate_target_next_day_return(df: pd.DataFrame) -> pd.DataFrame:
    return df['adj_close'].shift(-1) - df['adj_close']


def calculate_target_next_day_return_rate(df: pd.DataFrame) -> pd.DataFrame:
    return df['target_return_next_day'] / df['adj_close']


def calculate_target_next_day_label(df: pd.DataFrame) -> pd.DataFrame:
    return classify_three_way(df['target_return_rate_next_day'])


def calculate_target_next_3d_return(df: pd.DataFrame) -> pd.DataFrame:
    return df['adj_close'].shift(-3) - df['adj_close']


def calculate_target_next_3d_return_rate(df: pd.DataFrame) -> pd.DataFrame:
    return df['target_return_next_3d'] / df['adj_close']


def calculate_target_next_3d_label(df: pd.DataFrame) -> pd.DataFrame:
    return classify_three_way(df['target_return_rate_next_3d'])


def calculate_body_length(df: pd.DataFrame) -> pd.DataFrame:
    return abs(df['open'] - df['close'])

def calculate_upper_shadow(df: pd.DataFrame) -> pd.DataFrame:
    return df['high'] - np.maximum(df['open'], df['close'])

def calculate_lower_shadow(df: pd.DataFrame) -> pd.DataFrame:
    return np.minimum(df['open'], df['close']) - df['low']

def calculate_return_rate_n(df: pd.DataFrame, n: int) -> pd.Series:
    """N-day return rate based on close price."""
    return df["close"].pct_change(n)

def calculate_rsi_14(df: pd.DataFrame) -> pd.Series:
    """14-day RSI based on close price."""
    close = df["close"]
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(df: pd.DataFrame) -> pd.Series:
    """MACD line (EMA12 - EMA26) based on close price."""
    close = df["close"]
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    return ema12 - ema26

def calculate_ma_ratio_5_21(df: pd.DataFrame) -> pd.Series:
    """MA5 / MA21 ratio based on close price."""
    close = df["close"]
    ma5 = close.rolling(5, min_periods=1).mean()
    ma21 = close.rolling(21, min_periods=1).mean()
    return ma5 / ma21
