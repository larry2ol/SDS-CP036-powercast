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