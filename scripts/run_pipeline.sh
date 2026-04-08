#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE_RES="$ROOT/pipeline_res"
mkdir -p "$PIPELINE_RES"

echo "=== AAPL Trend Prediction Pipeline ==="
echo

PS3="Select training type: "
select MODE in "fixed_eps_0.005" "quantile" "quit"; do
  case "$MODE" in
    fixed_eps_0.005) MODE_KEY="fixed"; break ;;
    quantile) MODE_KEY="quantile"; break ;;
    quit) exit 0 ;;
    *) echo "Invalid choice." ;;
  esac
done

echo
OPTIONS=(
  "baselines_random_momentum_ma"
  "ridge"
  "logreg_multinomial_ovr"
  "svm_linear"
  "rf5_rf6"
  "rf_best_fast"
  "rf_best_slow"
  "gb"
  "hgb_best_fast"
  "lgbm"
  "ensemble_lr_svm"
  "quit"
)

echo "Select model(s) by number (space-separated):"
for i in "${!OPTIONS[@]}"; do
  printf "%2d) %s\n" "$((i+1))" "${OPTIONS[$i]}"
done

read -r -p "Model selection: " MODEL_INPUT

MODELS=()
for tok in $MODEL_INPUT; do
  if ! [[ "$tok" =~ ^[0-9]+$ ]]; then
    echo "Invalid input: $tok"
    exit 1
  fi
  idx=$((tok-1))
  if (( idx < 0 || idx >= ${#OPTIONS[@]} )); then
    echo "Invalid selection: $tok"
    exit 1
  fi
  if [[ "${OPTIONS[$idx]}" == "quit" ]]; then
    exit 0
  fi
  MODELS+=("${OPTIONS[$idx]}")
done

if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "No model selected."
  exit 1
fi

read -r -p "Run evaluation after training? (y/n): " RUN_EVAL
RUN_EVAL="$(printf "%s" "$RUN_EVAL" | tr '[:upper:]' '[:lower:]')"

read -r -p "Update reports/ files? (y/n): " UPDATE_REPORTS
UPDATE_REPORTS="$(printf "%s" "$UPDATE_REPORTS" | tr '[:upper:]' '[:lower:]')"

if [[ "$MODE_KEY" == "fixed" ]]; then
  TRAIN_DIR="$ROOT/src/train"
  RESULTS_DIR="$ROOT/results"
  REPORTS_DIR="$ROOT/reports"
  EVAL_CMD=(python "$ROOT/src/evaluate.py")
  SUFFIX=""
else
  TRAIN_DIR="$ROOT/src/train/quantile"
  RESULTS_DIR="$ROOT/results/quantile"
  REPORTS_DIR="$ROOT/reports/quantile"
  EVAL_CMD=(python "$ROOT/src/train/quantile/evaluate_quantile.py")
  SUFFIX="_quantile"
fi

set_model() {
  SCRIPT_CMD=()
  PRED_FILES=()
  EXTRA_FILES=()
  build_cmd() {
    local base="$1"; shift
    if [[ "$MODE_KEY" == "quantile" ]]; then
      SCRIPT_CMD=(env PYTHONPATH="$ROOT" python -m "src.train.quantile.${base}_quantile" "$@")
    else
      SCRIPT_CMD=(python "$TRAIN_DIR/${base}.py" "$@")
    fi
  }
  case "$1" in
    baselines_random_momentum_ma)
      build_cmd "train_baselines"
      PRED_FILES=("pred_random.csv" "pred_momentum.csv" "pred_ma.csv")
      ;;
    ridge)
      build_cmd "train_ridge"
      PRED_FILES=("pred_ridge.csv")
      ;;
    logreg_multinomial_ovr)
      build_cmd "train_logreg"
      PRED_FILES=("pred_logreg_multinomial.csv" "pred_logreg_ovr.csv")
      ;;
    svm_linear)
      build_cmd "train_svm"
      PRED_FILES=("pred_svm_linear.csv")
      ;;
    rf5_rf6)
      build_cmd "train_tree_models"
      PRED_FILES=("pred_rf5.csv" "pred_rf6.csv")
      ;;
    rf_best_fast)
      build_cmd "train_rf_search" --mode fast
      PRED_FILES=("pred_rf_best_fast.csv")
      EXTRA_FILES=("rf_grid_search_fast.csv")
      ;;
    rf_best_slow)
      build_cmd "train_rf_search" --mode slow
      PRED_FILES=("pred_rf_best_slow.csv")
      EXTRA_FILES=("rf_grid_search_slow.csv")
      ;;
    gb)
      build_cmd "train_gb"
      PRED_FILES=("pred_gb.csv")
      ;;
    hgb_best_fast)
      build_cmd "train_hgb_search" --mode fast
      PRED_FILES=("pred_hgb_best_fast.csv")
      EXTRA_FILES=("hgb_grid_search_fast.csv")
      ;;
    lgbm)
      build_cmd "train_lightgbm"
      PRED_FILES=("pred_lgbm.csv")
      ;;
    ensemble_lr_svm)
      build_cmd "train_ensemble"
      PRED_FILES=("pred_ensemble.csv")
      ;;
  esac
}

eval_methods_for_model() {
  case "$1" in
    baselines_random_momentum_ma) echo "random,momentum,ma" ;;
    ridge) echo "ridge" ;;
    logreg_multinomial_ovr) echo "logreg_multinomial,logreg_ovr" ;;
    svm_linear) echo "svm_linear" ;;
    rf5_rf6) echo "rf5,rf6" ;;
    rf_best_fast) echo "rf_best_fast" ;;
    rf_best_slow) echo "rf_best_slow" ;;
    gb) echo "gb" ;;
    hgb_best_fast) echo "hgb_best_fast" ;;
    lgbm) echo "lgbm" ;;
    ensemble_lr_svm) echo "ensemble_lr_svm" ;;
    *) echo "" ;;
  esac
}

timestamp="$(date +"%Y%m%d_%H%M%S")"
OUT_BASE="$PIPELINE_RES/${timestamp}_${MODE_KEY}"
mkdir -p "$OUT_BASE"

{
  echo "timestamp: $timestamp"
  echo "mode: $MODE_KEY"
  echo "models: ${MODELS[*]}"
  echo "evaluate: $RUN_EVAL"
} > "$OUT_BASE/options.txt"

for MODEL in "${MODELS[@]}"; do
  SCRIPT_CMD=()
  PRED_FILES=()
  EXTRA_FILES=()
  set_model "$MODEL"
  OUT_DIR="$OUT_BASE/$MODEL"
  mkdir -p "$OUT_DIR"

  {
    echo "timestamp: $timestamp"
    echo "mode: $MODE_KEY"
    echo "model: $MODEL"
    echo "script: ${SCRIPT_CMD[*]-}"
    echo "evaluate: $RUN_EVAL"
    echo "pred_files: ${PRED_FILES[*]-}"
    echo "extra_files: ${EXTRA_FILES[*]-}"
  } > "$OUT_DIR/options.txt"

  echo
  echo "Running: ${SCRIPT_CMD[*]}" | tee -a "$OUT_DIR/run.log"
  "${SCRIPT_CMD[@]}" 2>&1 | tee -a "$OUT_DIR/run.log"

  if [[ "$RUN_EVAL" == "y" || "$RUN_EVAL" == "yes" ]]; then
    echo | tee -a "$OUT_DIR/run.log"
    echo "Running evaluation: ${EVAL_CMD[*]}" | tee -a "$OUT_DIR/run.log"
    METHODS="$(eval_methods_for_model "$MODEL")"
    if [[ -n "$METHODS" ]]; then
      if [[ "$UPDATE_REPORTS" == "y" || "$UPDATE_REPORTS" == "yes" ]]; then
        env PIPELINE_METHODS="$METHODS" "${EVAL_CMD[@]}" 2>&1 | tee -a "$OUT_DIR/run.log"
      else
        env PIPELINE_METHODS="$METHODS" PIPELINE_NO_REPORTS=1 "${EVAL_CMD[@]}" 2>&1 | tee -a "$OUT_DIR/run.log"
      fi
    else
      if [[ "$UPDATE_REPORTS" == "y" || "$UPDATE_REPORTS" == "yes" ]]; then
        "${EVAL_CMD[@]}" 2>&1 | tee -a "$OUT_DIR/run.log"
      else
        env PIPELINE_NO_REPORTS=1 "${EVAL_CMD[@]}" 2>&1 | tee -a "$OUT_DIR/run.log"
      fi
    fi
  fi

  echo | tee -a "$OUT_DIR/run.log"
  echo "Collecting outputs..." | tee -a "$OUT_DIR/run.log"
  for f in "${PRED_FILES[@]}"; do
    if [[ -f "$RESULTS_DIR/$f" ]]; then
      cp "$RESULTS_DIR/$f" "$OUT_DIR/"
    fi
  done
  for f in "${EXTRA_FILES[@]}"; do
    if [[ -f "$RESULTS_DIR/$f" ]]; then
      cp "$RESULTS_DIR/$f" "$OUT_DIR/"
    fi
  done

  if [[ "$RUN_EVAL" == "y" || "$RUN_EVAL" == "yes" ]] && [[ "$UPDATE_REPORTS" == "y" || "$UPDATE_REPORTS" == "yes" ]]; then
    for f in evaluation_accuracy.csv evaluation_score.csv evaluation_detail.txt; do
      if [[ -f "$REPORTS_DIR/$f" ]]; then
        cp "$REPORTS_DIR/$f" "$OUT_DIR/"
      fi
    done
  fi

  echo "Done: $MODEL" | tee -a "$OUT_DIR/run.log"
  echo "Results saved to: $OUT_DIR" | tee -a "$OUT_DIR/run.log"
done
