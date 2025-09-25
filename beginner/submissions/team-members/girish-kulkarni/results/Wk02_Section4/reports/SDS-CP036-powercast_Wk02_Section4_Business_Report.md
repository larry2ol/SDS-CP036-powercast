# Week 2 — Section 4: Data Splitting & Preparation

**Primary input:** `scaled_standard_train.csv + scaled_standard_test.csv`
**Rows: 52416 | Train rows: 41932 | Test rows: 10484 | Target: Total_auto**

## Key Questions Answered
### 4. Data Splitting & Preparation
Q: How did you split your data into training and test sets to maintain chronological order?
A: We respected the **natural flow of time** in the data. If training/testing files already existed from Section 3, we used them as-is. Otherwise, we split by time: the **first 80% (earlier dates)** for training and the **last 20% (later dates)** for testing, with **no shuffling**. This matches real-world usage, where yesterday teaches us to predict tomorrow.

Q: What steps did you take to prevent information leakage between splits?
A: We prevented **information leakage**—that’s when future knowledge accidentally sneaks into training—by ensuring any learned settings (such as scaling/normalization from earlier steps) were **fit on the training period only** and then **applied to the later test period**. When building targets and features, we also avoided using any future-looking information. This keeps evaluation realistic for live operations.

Q: How did you verify that your train/test split was appropriate for time-series forecasting?
A: We validated the split in two ways: (1) a **timeline visualization** with a clear marker where training ends and testing begins; and (2) **walk‑forward validation**, which repeatedly tests on successive future slices. Think of it as asking, *“If we stopped here, could we predict the next step?”* Consistent results across slices and a clean split line indicate the split is sound for forecasting.

## Artifacts
- Train: `features/train.csv`
- Test: `features/test.csv`
- Walk-forward results: `features/walk_forward_results.csv`
- Plot: `wk02_section4_split_marker.png`
- Machine-readable summary: `summary.json`