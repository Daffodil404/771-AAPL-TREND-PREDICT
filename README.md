# AAPL Trend Prediction (Project Sketch)

## Goal
Predict next-day trend for AAPL as 3 classes: `up`, `down`, `flat`.

## Scope (from proposal)
- Data preprocessing
- EDA
- Baselines: Random, Momentum, Moving Average
- Simple model: Logistic Regression
- Evaluation: Accuracy + confusion matrix (+ precision/recall/F1)

## Folder Structure
- `data/raw/`: original downloaded data
- `data/processed/`: cleaned and feature-ready data
- `notebooks/`: EDA and quick experiments
- `src/`: scripts for pipeline steps
- `reports/figures/`: generated plots
- `results/`: metrics and prediction outputs

## Suggested Execution Order
1. `src/preprocess.py`
2. `src/eda.py`
3. `src/features.py`
4. `src/train_baselines.py`
5. `src/train_logreg.py`
6. `src/evaluate.py`

## Immediate Next Steps
1. Decide the `flat` threshold (e.g. `abs(return) < 0.5%`).
2. Download AAPL historical OHLCV data into `data/raw/`.
3. Implement preprocessing and generate `data/processed/aapl_clean.csv`.
