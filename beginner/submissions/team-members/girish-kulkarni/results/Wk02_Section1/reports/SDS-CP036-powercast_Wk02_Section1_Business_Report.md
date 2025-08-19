# Week 2 — Section 1: Time-Based Feature Engineering

**Dataset:** `Tetuan City power consumption.csv`
**Period: 2017-01-01 00:00:00 → 2017-12-30 23:50:00 | Rows: 52416 | Median step: 600 seconds**

1. Time-Based Feature Engineering
Q: Which time-based features did you create (e.g., hour, weekday, weekend, month), and why did you select them?
A: We created hour, day-of-week (and weekend flag), month, and quarter features, plus encodings that treat time as a circle so midnight is next to 11 PM. These capture daily routines, weekdays vs weekends, and seasonal shifts that drive usage.

Q: How did these new features help capture patterns in power consumption?
A: These features revealed consistent daily and weekly patterns—e.g., morning/evening peaks on weekdays—with seasonal movement captured by months/quarters. Cyclical encodings help models learn smooth curves across the day/week.

Q: Did you encounter any challenges when extracting or encoding time features? How did you address them?
A: We validated timestamps (including alternate date formats) and checked the typical time step. Where intervals were irregular, we highlighted those areas and relied on cyclical encodings to reduce edge effects.

## Artifacts
- Engineered dataset: `features/engineered_time_features.csv`
- Plots: ['wk02_section1_hourly_profile.png', 'wk02_section1_dayofweek_profile.png']
- Machine-readable summary: `summary.json`