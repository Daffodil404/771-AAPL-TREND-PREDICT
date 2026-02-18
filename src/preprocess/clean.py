"""Preprocess AAPL raw data into a clean time-indexed dataset."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repo root (has src/ and data/)
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def resolve_raw_path() -> Path:
    for name in ("AAPL.csv", "aapl.csv"):
        candidate = RAW_DIR / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No raw AAPL file found in {RAW_DIR}")


def load_aapl_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_aapl_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip().lower().replace(" ", "_") for c in out.columns]

    required_columns = ["date", "open", "high", "low", "close", "adj_close", "volume"]
    missing_columns = [c for c in required_columns if c not in out.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    out["date"] = pd.to_datetime(out["date"], dayfirst=True, errors="coerce")

    numeric_columns = ["open", "high", "low", "close", "adj_close", "volume"]
    for col in numeric_columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.drop_duplicates(subset=["date"], keep="last")
    out = out.sort_values("date").set_index("date")
    out = out.dropna(subset=required_columns[1:])
    out = out.loc[~out.index.isna()]
    return out


def main() -> None:
    raw_path = resolve_raw_path()
    print(f"Loading raw data from: {raw_path}")

    raw_data = load_aapl_data(raw_path)
    print(f"Raw shape: {raw_data.shape}")

    cleaned = clean_aapl_data(raw_data)
    print(f"Cleaned shape: {cleaned.shape}")
    print(f"Date range: {cleaned.index.min().date()} -> {cleaned.index.max().date()}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "aapl_clean.csv"
    cleaned.reset_index().to_csv(output_path, index=False)
    print(f"Saved cleaned data to: {output_path}")


if __name__ == "__main__":
    main()
