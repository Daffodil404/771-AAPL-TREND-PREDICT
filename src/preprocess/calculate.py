import pandas as pd
import numpy as np
from pathlib import Path
# Label Define, eps is 0.002
# 1. Up: Close > Open + eps
# 2. Down: Close < Open - eps
# 3. Flat: |Close - Open| < eps

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repo root (has src/ and data/)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

EPS = 0.002

def calculate_daily_range(df: pd.DataFrame) -> pd.DataFrame:
    return (df['high'] - df['low']) / df['close']

def calculate_intraday_returns(df: pd.DataFrame) -> pd.DataFrame:
    return df['close'] - df['open']

def calculate_intraday_return_rate(df: pd.DataFrame) -> pd.DataFrame:
    return df['intraday_return'] / df['open']

def calculate_intraday_labels(df: pd.DataFrame) -> pd.DataFrame:
    return np.where(df['intraday_return_rate'] > EPS, 'up', np.where(df['intraday_return_rate'] < -EPS, 'down', 'flat'))

def calculate_daily_return(df: pd.DataFrame) -> pd.DataFrame:
    return df['close'] - df['close'].shift(1)

def calculate_daily_return_rate(df: pd.DataFrame) -> pd.DataFrame:
    return df['daily_return'] / df['close'].shift(1)

def calculate_daily_labels(df: pd.DataFrame) -> pd.DataFrame:
    return np.where(df['daily_return_rate'] > EPS, 'up', np.where(df['daily_return_rate'] < -EPS, 'down', 'flat'))

def calculate_overnight_return(df: pd.DataFrame) -> pd.DataFrame:
    return df['open'] - df['close'].shift(1)

def calculate_overnight_return_rate(df: pd.DataFrame) -> pd.DataFrame:
    return df['overnight_return'] / df['close'].shift(1)

def calculate_overnight_labels(df: pd.DataFrame) -> pd.DataFrame:
    return np.where(df['overnight_return_rate'] > EPS, 'up', np.where(df['overnight_return_rate'] < -EPS, 'down', 'flat'))

# Volume is the number of shares traded

def calculate_volume_5(df: pd.DataFrame) -> pd.DataFrame:
    return df['volume'].rolling(window=5).mean()

def calculate_volume_10(df: pd.DataFrame) -> pd.DataFrame:
    return df['volume'].rolling(window=10).mean()

def calculate_volume_20(df: pd.DataFrame) -> pd.DataFrame:
    return df['volume'].rolling(window=20).mean()

# Volatility is the standard deviation of the daily return rates
def calculate_volatility_5(df: pd.DataFrame) -> pd.DataFrame:
    return df['daily_return_rate'].rolling(window=5).std()

def calculate_volatility_10(df: pd.DataFrame) -> pd.DataFrame:
    return df['daily_return_rate'].rolling(window=10).std()

def calculate_volatility_20(df: pd.DataFrame) -> pd.DataFrame:
    return df['daily_return_rate'].rolling(window=20).std()

def output_data(df: pd.DataFrame, output_path: Path) -> None:
    out = df.copy()
    out['intraday_return'] = calculate_intraday_returns(out)
    out['intraday_return_rate'] = calculate_intraday_return_rate(out)
    out['label'] = calculate_intraday_labels(out)
    out['daily_return'] = calculate_daily_return(out)
    out['daily_return_rate'] = calculate_daily_return_rate(out)
    out['daily_label'] = calculate_daily_labels(out)
    out['daily_range'] = calculate_daily_range(out)
    out['overnight_return'] = calculate_overnight_return(out)
    out['overnight_return_rate'] = calculate_overnight_return_rate(out)
    out['overnight_label'] = calculate_overnight_labels(out)
    out['volume_5'] = calculate_volume_5(out)
    out['volume_10'] = calculate_volume_10(out)
    out['volume_20'] = calculate_volume_20(out)
    out['volatility_5'] = calculate_volatility_5(out)
    out['volatility_10'] = calculate_volatility_10(out)
    out['volatility_20'] = calculate_volatility_20(out)
    out.to_csv(output_path, index=False)

def load_data(input_path: Path) -> pd.DataFrame:
    return pd.read_csv(input_path)

def main() -> None:
    input_path = PROCESSED_DIR / "aapl_clean.csv"
    df = load_data(input_path)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "aapl_processed.csv"
    output_data(df, output_path)

if __name__ == "__main__":
    main()