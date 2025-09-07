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