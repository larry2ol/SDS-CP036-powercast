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