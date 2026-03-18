import pandas as pd

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

def calculate_volatility_30(df: pd.DataFrame) -> pd.DataFrame:
    return df['daily_return_rate'].rolling(window=30).std()

