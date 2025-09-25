# SDS-CP036-powercast - Week 3 Section 2: MLflow Experiment Tracking

Profile: **dev**

## Key Questions Answered

Q: Which evaluation metrics did you use to assess model performance, and why are they appropriate for this problem?
A: We used RMSE, MAE, and MAPE. RMSE penalizes large errors (useful for risk), MAE provides an easy-to-explain average miss, and MAPE gives a percentage error that business users understand.

Q: How did you use MLflow to track your experiments and results?
A: We created an MLflow experiment named powercast_wk03_s2 and logged, for each run:
- Parameters: model name, resampling (hour/hourly), lags, seasonal order, profile tag, fast_mode flag.
- Metrics: RMSE, MAE, MAPE for the evaluation window.
- Artifacts: Actual vs Predicted plots per zone (in single hold-out), or per fold (in backtesting).
Tracking runs locally via a file-based MLflow backend keeps everything versioned inside the repo (mlruns/).

Q: What insights did you gain from comparing actual vs. predicted curves for each zone?
A: Visual comparisons showed how models capture shape (daily patterns), timing (peaks/valleys), and magnitude (over/under-bias). These help pinpoint which model is most reliable for each zone's operations.

[Run Summary - CSV](mlflow_run_summary.csv)

---

## Business Value Summary (Executive View)
- Transparency & Trust: Every experiment is versioned with parameters, metrics, and plots so leaders can see how conclusions were reached.
- Faster Decisions: Side-by-side comparisons shorten the path to a "champion" model for each zone.
- Risk Control: Using RMSE/MAE/MAPE together prevents over-optimizing for a single metric and missing real-world errors.
- Governance & Auditability: A local MLflow history (mlruns/) provides an auditable trail for compliance, board reviews, or customer assurances.
- Scalability: Profiles (dev/preprod/final) let us move from quick smoke-tests to robust selection without changing code.

---

# SDS-CP036-powercast - Week 3 Section 1: Model Selection & Training

Profile: **dev**

## Key Questions Answered

Q: Which machine learning models did you choose for forecasting power consumption, and what motivated your selections?
A: We compared a simple Baseline (naive), a statistical time-series model (SARIMAX) that handles daily seasonality, an optional Prophet for trend/seasonality decomposition, and an optional XGBoost model with lag features for non-linear patterns.

Q: How did you structure your models to handle the multi-zone prediction task (separate models vs. multi-output)?
A: We trained separate models for each zone (Zone 1, Zone 2, Zone 3). This avoids cross-zone interference and keeps insights clear for operations teams.

Q: What challenges did you encounter during model training, and how did you address them?
A: Runtime and data quirks were the main issues. We used hourly resampling and capped history for speed; we normalized column headers to remove double spaces; and we set lighter SARIMAX and XGBoost defaults for fast, reliable runs. Prophet is optional and limited in dev to keep execution snappy.

[Model Comparison - CSV](model_comparison.csv)

---

## Business Value Summary (Executive View)
- Faster iteration: Profiles (dev/preprod/final) let us move from quick smoke-tests to rigorous selection without changing code.
- Clear decisions: Side-by-side metrics identify the best model per zone; the final profile adds fold averages and stability checks.
- Reduced risk: Using RMSE/MAE/MAPE together prevents optimizing for a single number that might miss operational errors.
- Transparency: Reproducible splits, consistent resampling, and saved artifacts make results easy to explain to stakeholders.
- Scalability: The per-zone approach and shared pipeline scale cleanly as new data or zones are added.

---

---

# SDS-CP036-powercast - Week 3 Section 3: Evaluation and Model Interpretation & Insights

Profile: **dev**

## Key Questions Answered

Q: How did you interpret feature importance or model coefficients, and what did they reveal about power consumption drivers?
A: We used **two lenses**:
- **Correlations with weather/solar inputs** (e.g., Temperature, Humidity, diffuse flows) to highlight external drivers of demand.
- **Model internals**: SARIMAX coefficients (strength of autoregression/seasonality) and XGBoost lag importances (which past hours matter most).
Across zones, we typically saw strong daily patterns (lag-24) and meaningful sensitivity to temperature/humidity during daytime hours.

Q: Did you observe any systematic errors or biases in your model predictions? How did you investigate and address them?
A: We examined **residuals by hour-of-day**. Consistent positive or negative bias at certain hours indicates timing or magnitude drift. Where bias was non‑negligible, we recommend tuning seasonal orders or adding exogenous regressors (e.g., temperature) or holiday indicators to reduce recurring misfit.

Q: What trade-offs did you consider when selecting your final model(s) for each zone?
A: Trade-offs balanced **accuracy vs. simplicity vs. speed**. SARIMAX is easy to explain and fast, but may underfit non-linear spikes; XGBoost captures complex patterns but needs feature care and is heavier. We keep **per-zone** models for clarity and operational accountability.

[Feature Correlations - CSV](feature_correlations.csv)
[Model Importances - CSV](model_importances.csv)
[Residual Bias by Hour - CSV](residual_bias_by_hour.csv)

---

## Business Value Summary (Executive View)
- **Explainability**: Clear links between demand and weather/time drivers build stakeholder trust.
- **Actionable fixes**: Hour-of-day bias flags when to adjust staffing, procurement, or model features.
- **Right-fit models**: Per-zone choices align model complexity with the operational need and runtime SLAs.
- **Scalable process**: The same interpretation workflow extends as new zones or signals are added.
