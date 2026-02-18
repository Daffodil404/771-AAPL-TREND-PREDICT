import pandas as pd
import numpy as np

EPS = 0.002


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


def calculate_body_length(df: pd.DataFrame) -> pd.DataFrame:
    return abs(df['open'] - df['close'])

def calculate_upper_shadow(df: pd.DataFrame) -> pd.DataFrame:
    return df['high'] - np.maximum(df['open'], df['close'])

def calculate_lower_shadow(df: pd.DataFrame) -> pd.DataFrame:
    return np.minimum(df['open'], df['close']) - df['low']
