import pandas as pd
import numpy as np
from pathlib import Path
from base import *
from rolling import *
# Label rule (eps = 0.005):
# up: return_rate > eps
# down: return_rate < -eps
# flat: otherwise

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # repo root (src/preprocess/calculate/ -> 3 up)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def output_data(df: pd.DataFrame, output_path: Path) -> None:
    out = df.copy()
    out['intraday_return'] = calculate_intraday_returns(out)
    out['intraday_return_rate'] = calculate_intraday_return_rate(out)
    out['intraday_label'] = calculate_intraday_labels(out)
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
    out['volatility_30'] = calculate_volatility_30(out)
    out['body_length'] = calculate_body_length(out)
    out['upper_shadow_length'] = calculate_upper_shadow(out)
    out['lower_shadow_length'] = calculate_lower_shadow(out)
    out['return_3'] = calculate_return_rate_n(out, 3)
    out['return_5'] = calculate_return_rate_n(out, 5)
    out['return_10'] = calculate_return_rate_n(out, 10)
    out['rsi_14'] = calculate_rsi_14(out)
    out['macd'] = calculate_macd(out)
    out['ma_ratio_5_21'] = calculate_ma_ratio_5_21(out)
    out['return_3_volatility_5'] = out['return_3'] * out['volatility_5']

    # Main supervised target: next-day adjusted-close-to-adjusted-close trend.
    out['target_return_next_day'] = calculate_target_next_day_return(out)
    out['target_return_rate_next_day'] = calculate_target_next_day_return_rate(out)
    out['target_label_next_day'] = calculate_target_next_day_label(out)
    out['target_return_next_3d'] = calculate_target_next_3d_return(out)
    out['target_return_rate_next_3d'] = calculate_target_next_3d_return_rate(out)
    out['target_label_next_3d'] = calculate_target_next_3d_label(out)

    # Keep `label` as the default training target for compatibility.
    out['label'] = out['target_label_next_day']
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
