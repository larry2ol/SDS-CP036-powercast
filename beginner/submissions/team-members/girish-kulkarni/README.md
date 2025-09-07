# PowerCast – Project README

_Generated/updated: 2025-08-23 06:15:08_

## Week 1 – Key Notebooks & Reports

### Time Consistency & Structure
- Notebook: `GirishK_PwrCst_Wk1_Section1-Business.ipynb`
- Purpose: Checks timestamp integrity (hourly grid), gaps/overlaps, duplicates.
- Expected reports (by profile):
  - `dev` report: `results/Wk01_Section1_dev/reports/SDS-CP036-powercast_Wk01_Section1_Business_Report.md`
  - `preprod` report: `results/Wk01_Section1_preprod/reports/SDS-CP036-powercast_Wk01_Section1_Business_Report.md`
  - `final` report: `results/Wk01_Section1_final/reports/SDS-CP036-powercast_Wk01_Section1_Business_Report.md`

### Temporal Trends & Seasonality
- Notebook: `GirishK_PwrCst_Wk1_Section2-Business.ipynb`
- Purpose: Daily/weekly patterns via profiles, heatmaps, rolling trends.
- Expected reports (by profile):
  - `dev` report: `results/Wk01_Section2_dev/reports/SDS-CP036-powercast_Wk01_Section2_Business_Report.md`
  - `preprod` report: `results/Wk01_Section2_preprod/reports/SDS-CP036-powercast_Wk01_Section2_Business_Report.md`
  - `final` report: `results/Wk01_Section2_final/reports/SDS-CP036-powercast_Wk01_Section2_Business_Report.md`

### Environmental Feature Relationships
- Notebook: `GirishK_PwrCst_Wk1_Section3-Business.ipynb`
- Purpose: Correlation scans with Temperature, Humidity, Wind, Diffuse flows.
- Expected reports (by profile):
  - `dev` report: `results/Wk01_Section3_dev/reports/SDS-CP036-powercast_Wk01_Section3_Business_Report.md`
  - `preprod` report: `results/Wk01_Section3_preprod/reports/SDS-CP036-powercast_Wk01_Section3_Business_Report.md`
  - `final` report: `results/Wk01_Section3_final/reports/SDS-CP036-powercast_Wk01_Section3_Business_Report.md`

### Lag Effects & Time Dependency
- Notebook: `GirishK_PwrCst_Wk1_Section4-Business.ipynb`
- Purpose: Lag scouting (lag‑1, lag‑24) using lag corr, ACF/PACF.
- Expected reports (by profile):
  - `dev` report: `results/Wk01_Section4_dev/reports/SDS-CP036-powercast_Wk01_Section4_Business_Report.md`
  - `preprod` report: `results/Wk01_Section4_preprod/reports/SDS-CP036-powercast_Wk01_Section4_Business_Report.md`
  - `final` report: `results/Wk01_Section4_final/reports/SDS-CP036-powercast_Wk01_Section4_Business_Report.md`

### Data Quality & Sensor Anomalies
- Notebook: `GirishK_PwrCst_Wk1_Section5-Business.ipynb`
- Purpose: Outlier flags (negatives, spikes) and quality audit.
- Expected reports (by profile):
  - `dev` report: `results/Wk01_Section5_dev/reports/SDS-CP036-powercast_Wk01_Section5_Business_Report.md`
  - `preprod` report: `results/Wk01_Section5_preprod/reports/SDS-CP036-powercast_Wk01_Section5_Business_Report.md`
  - `final` report: `results/Wk01_Section5_final/reports/SDS-CP036-powercast_Wk01_Section5_Business_Report.md`

### Consolidated Week 1 Report
- `SDS-CP036-powercast_Wk01_Report_Business.md`

---

## Week 2 – Key Notebooks & Reports

### Time-Based Feature Engineering
- Notebook: `GirishK_PwrCst_Wk2_Section1-Business.ipynb`
- Purpose: Hour/weekday/weekend/month features (+ optional cyclical encodings).
- Expected reports (by profile):
  - `dev` report: `results/Wk02_Section1_dev/reports/SDS-CP036-powercast_Wk02_Section1_Business_Report.md`
  - `preprod` report: `results/Wk02_Section1_preprod/reports/SDS-CP036-powercast_Wk02_Section1_Business_Report.md`
  - `final` report: `results/Wk02_Section1_final/reports/SDS-CP036-powercast_Wk02_Section1_Business_Report.md`

### Lag & Rolling Statistics
- Notebook: `GirishK_PwrCst_Wk2_Section2-Business.ipynb`
- Purpose: Lag‑1/lag‑24 and short rolling stats for stability.
- Expected reports (by profile):
  - `dev` report: `results/Wk02_Section2_dev/reports/SDS-CP036-powercast_Wk02_Section2_Business_Report.md`
  - `preprod` report: `results/Wk02_Section2_preprod/reports/SDS-CP036-powercast_Wk02_Section2_Business_Report.md`
  - `final` report: `results/Wk02_Section2_final/reports/SDS-CP036-powercast_Wk02_Section2_Business_Report.md`

### Feature Scaling & Normalization
- Notebook: `GirishK_PwrCst_Wk2_Section3-Business.ipynb`
- Purpose: Scaling/normalization fit on train only (leakage‑safe).
- Expected reports (by profile):
  - `dev` report: `results/Wk02_Section3_dev/reports/SDS-CP036-powercast_Wk02_Section3_Business_Report.md`
  - `preprod` report: `results/Wk02_Section3_preprod/reports/SDS-CP036-powercast_Wk02_Section3_Business_Report.md`
  - `final` report: `results/Wk02_Section3_final/reports/SDS-CP036-powercast_Wk02_Section3_Business_Report.md`

### Data Splitting & Preparation
- Notebook: `GirishK_PwrCst_Wk2_Section4-Business.ipynb`
- Purpose: Chronological splits train→val→test; indices saved for audit.
- Expected reports (by profile):
  - `dev` report: `results/Wk02_Section4_dev/reports/SDS-CP036-powercast_Wk02_Section4_Business_Report.md`
  - `preprod` report: `results/Wk02_Section4_preprod/reports/SDS-CP036-powercast_Wk02_Section4_Business_Report.md`
  - `final` report: `results/Wk02_Section4_final/reports/SDS-CP036-powercast_Wk02_Section4_Business_Report.md`

### Data Quality & Preprocessing
- Notebook: `GirishK_PwrCst_Wk2_Section5-Business.ipynb`
- Purpose: Imputation/capping pipelines; stability checks across slices.
- Expected reports (by profile):
  - `dev` report: `results/Wk02_Section5_dev/reports/SDS-CP036-powercast_Wk02_Section5_Business_Report.md`
  - `preprod` report: `results/Wk02_Section5_preprod/reports/SDS-CP036-powercast_Wk02_Section5_Business_Report.md`
  - `final` report: `results/Wk02_Section5_final/reports/SDS-CP036-powercast_Wk02_Section5_Business_Report.md`

### Consolidated Week 2 Report
- `SDS-CP036-powercast_Wk02_Report_Business.md`

---

## Week 3 – Key Notebooks & Reports

### Model Selection & Training
- Notebook: `GirishK_PwrCst_Wk3_Section1-Business.ipynb`
- Purpose: Baseline vs SARIMAX vs XGBoost; metrics and A vs P plots per zone.
- Expected reports (by profile):
  - `dev` report: `results/Wk03_Section1_dev/reports/SDS-CP036-powercast_Wk03_Section1_Business_Report.md`
  - `preprod` report: `results/Wk03_Section1_preprod/reports/SDS-CP036-powercast_Wk03_Section1_Business_Report.md`
  - `final` report: `results/Wk03_Section1_final/reports/SDS-CP036-powercast_Wk03_Section1_Business_Report.md`

### MLflow Experiment Tracking & Evaluation
- Notebook: `GirishK_PwrCst_Wk3_Section2-Business.ipynb`
- Purpose: MLflow logging (params/metrics/artifacts); hold‑out/backtests.
- Expected reports (by profile):
  - `dev` report: `results/Wk03_Section2_dev/reports/SDS-CP036-powercast_Wk03_Section2_Business_Report.md`
  - `preprod` report: `results/Wk03_Section2_preprod/reports/SDS-CP036-powercast_Wk03_Section2_Business_Report.md`
  - `final` report: `results/Wk03_Section2_final/reports/SDS-CP036-powercast_Wk03_Section2_Business_Report.md`

### Evaluation & Model Interpretation & Insights
- Notebook: `GirishK_PwrCst_Wk3_Section3-Business.ipynb`
- Purpose: Correlations, SARIMAX coefs, XGBoost importances, residual bias.
- Expected reports (by profile):
  - `dev` report: `results/Wk03_Section3_dev/reports/SDS-CP036-powercast_Wk03_Section3_Business_Report.md`
  - `preprod` report: `results/Wk03_Section3_preprod/reports/SDS-CP036-powercast_Wk03_Section3_Business_Report.md`
  - `final` report: `results/Wk03_Section3_final/reports/SDS-CP036-powercast_Wk03_Section3_Business_Report.md`

### Consolidated Week 3 Report
- `SDS-CP036-powercast_Wk03_Report_Business.md`

---

## How to Run the Notebooks
1. Open the notebook in JupyterLab.
2. (Optional) Set `PROFILE = "dev" | "preprod" | "final"` at the top of the single cell (where supported).
3. Execute the single code cell.
4. Open the generated plots/CSVs and the Section report under the corresponding `results/Wk0X_SectionY_<profile>/reports/` folder.

**Profiles**
- `dev`: fast smoke test (1 zone, 7-day test window)
- `preprod`: fuller check (3 zones, 28-day test window)
- `final`: full run (3 zones, capped to last 365 days)

---
## Notes
- This generator is idempotent: re-running will refresh timestamps and paths.
- If any reports are missing, run the corresponding notebook first to generate them.
