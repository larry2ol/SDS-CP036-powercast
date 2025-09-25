# Week 2 — Section 3: Feature Scaling & Normalization

**Input dataset:** `engineered_lag_rolling_imputed.csv`
**Rows:** 52416 | **Period:** 2017-01-01 00:00:00 → 2017-12-30 23:50:00

## Key Questions Answered
### 3. Feature Scaling & Normalization
Q: Which normalization or scaling techniques did you apply to your numerical features, and why?
A: We prepared the numbers so different features are on comparable scales, which helps models learn reliably and keeps one large-magnitude feature from dominating the rest. We used three common approaches (each with a clear use case):
- **Standardization (z-score)** – centers at 0 and scales by typical spread (standard deviation). Good default when data is roughly bell-shaped.
- **Min–Max scaling** – compresses values to a 0–1 range. Helpful for algorithms sensitive to absolute ranges.
- **Robust scaling** – centers by the median and scales by the IQR (inter-quartile range); more resistant to outliers and skew.
Using all three gives the team flexibility to pick the best fit for downstream modeling without re-running feature engineering.

Q: How did you ensure that scaling was performed without introducing data leakage?
A: We guarded against **data leakage**—letting future information seep into training—by splitting the data **by time**, fitting each scaler **only on the training period**, and applying that scaling to the later **test period**. This mirrors production and keeps evaluation honest.

Q: Did you notice any features that required special treatment during normalization?
A: We intentionally **left some features unscaled** because they are binary flags (0/1) or already unitless cyclical encodings (sine/cosine in [-1,1]). Examples: cos_dow, cos_doy, cos_hour, is_weekend, sin_dow, sin_doy, sin_hour. For outlier‑prone metrics we prefer **Robust scaling** to avoid over‑weighting spikes.

## Artifacts
- Scaled features (Standard): `features/scaled_standard_train.csv`, `features/scaled_standard_test.csv`
- Scaled features (MinMax): `features/scaled_minmax_train.csv`, `features/scaled_minmax_test.csv`
- Scaled features (Robust): `features/scaled_robust_train.csv`, `features/scaled_robust_test.csv`
- Scaler parameters: `features/scalers.json`
- Plots: ['standard_Zone 1 Power Consumption_lag1_before.png', 'standard_Zone 1 Power Consumption_lag1_after.png']
- Machine-readable summary: `summary.json`