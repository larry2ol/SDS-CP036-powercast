# Week 2 — Section 2: Lag and Rolling Statistics

**Dataset:** `Tetuan City power consumption.csv`
**Period: 2017-01-01 00:00:00 → 2017-12-30 23:50:00 | Rows: 52416 | Median step: 600 seconds | Chained inputs: time features (Section 1)**

## Key Questions Answered
### 1. Lag and Rolling Statistics
Q: How did you determine which lag features and rolling statistics (mean, std, median, etc.) to engineer for each zone?
A: We designed the lag (historical shift) and rolling (moving average) features to reflect real-world usage rhythms. Instead of building features mechanically, we aligned them with how people actually consume electricity:
- Immediate past (one step back) captures short-term reactions (e.g., appliances switching on/off).
- About one hour back reflects near-term cycles at home or in offices.
- About one day back (24 hours) captures daily repetition in routines.
- Rolling averages and variability over ~1 hour, ~3 hours, ~24 hours smooth noise and expose typical levels and unusual spikes. We also merged time-based features from Section 1 where available.

Q: What impact did lag and rolling features have on model performance or interpretability?
A: These features made the model both more accurate and easier to explain. For accuracy: lag-1 reduced Mean Absolute Error (MAE — average absolute prediction error) by ~88.6% vs a simple mean baseline; the 1-hour rolling average reduced MAE by ~81.4% vs the same baseline. For interpretability: rolling windows smooth short-term volatility so planners can see the underlying daily pattern (e.g., typical morning/evening peaks).

Q: How did you handle missing values introduced by lag or rolling computations?
A: Lag/rolling features naturally create missing values at the start of the series. We kept those gaps in the raw engineered file for transparency. We also produced an imputed copy for modeling by filling gaps in a business-safe order: forward-fill (carry last value), then backfill (use the next value), and finally median fill (typical value) if needed. This keeps models robust without distorting early readings.

## Artifacts
- Engineered dataset (raw): `features/engineered_lag_rolling.csv`
- Engineered dataset (imputed): `features/engineered_lag_rolling_imputed.csv`
- Plots: ['wk02_section2_rolling24_overlay.png', 'wk02_section2_baseline_mae.png']
- Machine-readable summary: `summary.json`