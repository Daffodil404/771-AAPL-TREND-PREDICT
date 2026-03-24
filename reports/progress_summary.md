## AAPL 趋势预测项目进度总结

### 1. 数据与标注

- **数据范围**: AAPL 日频 OHLCV，从 1980-12-12 至 2022-03-24。
- **预处理结果**: `data/processed/aapl_processed.csv`，包含价格、成交量及一系列派生变量：
  - 日内/隔夜收益及收益率：`intraday_return_rate`, `overnight_return_rate` 等
  - 日收益及波动率：`daily_return_rate`, `volatility_5`, `volatility_10`, `volatility_20`
  - 成交量特征：`volume_5`, `volume_10`, `volume_20`
  - K 线形态特征：`body_length`, `upper_shadow_length`, `lower_shadow_length`
- **标签定义**: 预测目标为**下一交易日收盘价相对当日收盘价的方向**：
  - 使用 `daily_return_rate` 与阈值 `flat_threshold = 0.005`（约 0.5%）定义 `up / down / flat`
  - 主标签列：`target_label_next_day`（并在部分流程中保留别名 `label` 保持兼容）

### 2. EDA 与数据划分

- **整体标签分布**（见 `overall_target_label_distribution.png`）：
  - `up` / `down` 相比 `flat` 明显偏少，存在一定类别不平衡。
- **按年份的标签组成**（见 `yearly_target_label_composition.png`）：
  - 不同年份中 `up/down/flat` 的占比随着时间发生变化，说明存在明显的**时变结构 / regime 变化**。
- **时间序列划分**（见 `eda_time_split_overview.png`）：
  - 采用**时间顺序划分** train/val/test，避免信息泄露：
    - Train: 1980-12-12 ~ 2009-10-27（约 70%）
    - Val:   2009-10-28 ~ 2016-01-11（约 15%）
    - Test:  2016-01-12 ~ 2022-03-24（约 15%）
  - 划分在 `eda_extension.py` 与后续 `train_baselines.py` / `train_logreg.py` 中保持一致。

### 3. 特征工程

- 在 `src/features.py` 中，从 `aapl_processed.csv` 构造基础特征表 `data/processed/aapl_features.csv`。
- 当前用于建模的特征列：
  - `daily_return_rate`
  - `overnight_return_rate`
  - `volatility_5`
  - `volatility_10`
  - `volume_5`
  - `body_length`
- 输出列包括：`date`、上述 6 个特征以及目标列 `target_label_next_day`，并对含缺失值的样本行进行了过滤。

### 4. 基线方法（Baselines）

在 `src/train_baselines.py` 中实现并评估了 3 个基线方法，预测文件保存在 `results/`：

- **Random baseline**（`pred_random.csv`）
  - 依据训练集标签比例，随机抽样生成 `up/down/flat` 预测。
  - **Test accuracy**: ≈ **0.328**
- **Momentum baseline**（`pred_momentum.csv`）
  - 使用当日 `daily_return_rate` 的符号和阈值（与 EPS 一致）：
    - `> +0.005` → 预测 `up`
    - `< -0.005` → 预测 `down`
    - 其余 → 预测 `flat`
  - **Test accuracy**: ≈ **0.353**
- **Moving-average baseline**（`pred_ma.csv`）
  - 使用 `close` 的 5 日/20 日移动均线：
    - `MA5 > MA20` → `up`
    - `MA5 < MA20` → `down`
    - 其余 → `flat`
  - 在本实验中几乎从不预测 `flat`，但在 `up/down` 间有一定区分度。
  - **Test accuracy**: ≈ **0.370**

### 5. Logistic Regression 模型

在 `src/train_logreg.py` 中实现了带标准化与类别权重的三分类 Logistic Regression：

- **特征预处理**：
  - 使用 `StandardScaler` 对 6 个特征做标准化（仅在训练集上 `fit`，再对 train/test 做 `transform`），缓解不同量纲（如 `volume_5` 与收益率）带来的数值主导问题。
- **类别权重**：
  - 采用 `class_weight="balanced"`，由 sklearn 按类别频次自动计算权重，缓解 `up` 为多数类导致模型“总猜 up”的问题。
  - 尝试过手动调整权重（如提高 `down` 的权重），发现验证集准确率反而下降，最终回退到 `balanced` 方案。
- **模型形式**：
  - Multinomial LR：`LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")`
  - OvR LR：`OneVsRestClassifier(LogisticRegression(..., class_weight="balanced"))`
- **测试集表现（来自 `src/evaluate.py` 输出）**：
  - `logreg_multinomial`：
    - **Accuracy**: ≈ **0.415**
    - Confusion matrix 显示：
      - `up` 的 recall 较高（≈ 0.80），precision ≈ 0.41；
      - `flat` 有一定召回（≈ 0.32）；`down` 的召回偏低，但 precision 较高（约 0.40）。
  - `logreg_ovr`：
    - **Accuracy**: ≈ **0.410**
    - 整体略逊于 multinomial，但模式相似：更偏向识别 `up`，`down` 召回较低。

### 6. 与其它 baseline 的对比

- `src/evaluate.py` 汇总了各方法在同一 test 集上的表现（`reports/evaluation_accuracy.csv`）：
  - `logreg_multinomial` ≈ **0.415**
  - `logreg_ovr` ≈ **0.410**
  - `ma` ≈ **0.370**
  - `momentum` ≈ **0.353**
  - `random` ≈ **0.328**
- 额外计算的 **“与前一天一致”持久性 baseline**（预测明日标签 = 今日标签）：
  - 在 test 集上的 **accuracy ≈ 0.354**。
  - 因此，Logistic Regression 明显优于简单的“明日 = 今日”规则和其它 baselines。

### 7. 当前进度小结与后续方向

- **目前已完成**：
  - 数据预处理与多种价格/波动/成交量/K 线特征构造；
  - EDA 与时间序列划分可视化（标签分布、年份分布、train/val/test 划分图）；
  - 固定阈值（0.002/0.005/0.01）下的 `up/down/flat` 标签敏感性分析；
  - 基线方法（Random / Momentum / Moving-average）与三分类 Logistic Regression；
  - 统一的评估脚本，输出 accuracy 对比表与每种方法的混淆矩阵、分类报告。
- **潜在改进方向（可选）**：
  - 在 `features.py` 中增加滞后特征（lagged returns/volatility）、多周期收益（3/5/10 日）与价格/成交量相对均线特征；
  - 引入外部市场信息（纳斯达克指数收益、VIX、财报日期等），在 `features.py` 里按日期对齐后加入特征；
  - 尝试非线性模型（如 Random Forest / XGBoost）并通过验证集选择超参数；
  - 针对 `down` 类召回偏低的问题，进一步平衡 precision/recall（例如基于验证集调节 class_weight 或决策阈值），并在报告中展示 trade-off。

---

## AAPL 趋势预测项目进度总结（更新版，2026-03-22）

### 1. 数据与特征（新增/扩展）

- 新增外部市场数据：纳斯达克指数数据已对齐并保存于 `data/processed/nasdaq_ixic.csv`。
- 3 日收益版本特征：`src/features_3d.py` 生成 `data/processed/aapl_features_3d.csv`。
- 分位数版本特征：`src/features_quantile.py` 生成 `data/processed/quantile/aapl_features.csv`。

### 2. 模型体系扩展

- 树模型与集成方法已实现并批量训练：Random Forest、GBM、HGB、LightGBM、SVM、Ensemble。
- 对应训练脚本集中在 `src/train/*`，包括 `train_rf_search.py`、`train_tree_models.py`、`train_gb.py`、`train_hgb_search.py`、`train_lightgbm.py`、`train_svm.py`、`train_ensemble.py`。
- 预测输出集中于 `results/`，例如 `pred_rf_best_fast.csv`、`pred_gb.csv`、`pred_hgb_best_fast.csv`、`pred_lgbm.csv`、`pred_svm_linear.csv`、`pred_ensemble.csv`。

### 3. 变体实验（新增）

- **3 日收益预测任务**：独立训练管线 `src/train/3d/*`，结果与评估在 `results/3-day-return/` 与 `reports/3-day-return/`。
- **分位数预测任务**：独立训练管线 `src/train/quantile/*`，结果与评估在 `results/quantile/` 与 `reports/quantile/`。

### 4. 评估与对比（新增）

- 主任务评估汇总仍在 `reports/evaluation_accuracy.csv`、`reports/evaluation_detail.txt`、`reports/evaluation_score.csv`。
- 子阶段/分段评估已加入：`src/evaluate_subperiod.py` 输出 `reports/evaluation_subperiod.csv` 与 `reports/evaluation_subperiod.txt`。
- 3 日收益与分位数版本的评估报告：`reports/3-day-return/*` 与 `reports/quantile/*`。

### 5. 当前阶段成果小结（对比用）

- **阶段一（旧版总结）**：完成数据清洗、基础特征、EDA、基线方法与 Logistic Regression，并形成统一评估脚本。
- **阶段二（更新内容）**：扩展为多模型体系（树模型/GBM/HGB/LGBM/SVM/Ensemble），新增 3 日收益与分位数两套变体实验，并补充分段评估与更完整的结果产出目录。
