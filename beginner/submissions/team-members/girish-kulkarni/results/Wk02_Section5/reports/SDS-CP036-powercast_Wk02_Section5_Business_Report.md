# Week 2 — Section 5: Data Quality & Preprocessing

**Primary input:** `train.csv + test.csv`
**Rows (train): 41932 | Rows (test): 10484 | Duplicate timestamps removed (train/test): 0/0**

## Key Questions Answered
### 5. Data Quality & Preprocessing
Q: What preprocessing steps did you apply to handle missing values or anomalies before modeling?
A: We focused on making the dataset **trustworthy** before modeling. Specifically:
- **Missing values:** We filled short gaps using forward/backward fill; longer gaps were filled with a stable value learned from training (the **median**), and we logged where this occurred.
- **Anomalies:** We flagged unrealistic values (e.g., negative energy use or extreme spikes) using robust thresholds and treated them as missing so they wouldn’t skew results.
- **Duplicates & types:** We removed duplicate timestamps and ensured date/time formats and numeric types were consistent.

Q: How did you validate that your feature engineering and preprocessing pipeline produced consistent and reliable results across different data subsets?
A: We checked **pipeline stability** across different slices of the data. We compared missing‑value rates, anomaly counts, and feature ranges across subsets (e.g., earlier vs. later periods). Consistency across these checks tells us the preprocessing behaves reliably and won’t produce surprises in production.

## Artifacts
- Cleaned train: `features/cleaned_train.csv`
- Cleaned test: `features/cleaned_test.csv`
- Data quality summary: `features/data_quality_summary.json`
- Plots: ['missing_before_train.png', 'missing_after_train.png']
- Machine-readable summary: `summary.json`