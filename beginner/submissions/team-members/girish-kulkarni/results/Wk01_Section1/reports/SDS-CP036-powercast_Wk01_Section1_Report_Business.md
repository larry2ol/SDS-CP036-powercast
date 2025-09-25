# 💼 Week 1 – Section1: Time Consistency & Structure (Business-Friendly Report)

## Dataset
Using file: **Tetuan City power consumption.csv**

## Key Questions Answered
**Q1: Are there any missing or irregular timestamps in the dataset? How did you verify consistency?**  
I created a canonical `DateTime` column and inspected gaps between consecutive records to detect irregularities.

**Q2: What is the sampling frequency and are all records spaced consistently?**  
I measured the time deltas between consecutive rows. The most common spacing is **0 days 00:10:00**, suggesting the intended sampling cadence.

**Q3: Did you encounter any duplicates or inconsistent `DateTime` entries?**  
I found **0** duplicate timestamps (exact same `DateTime`). These could be reviewed or deduplicated based on your business rules.

## Plain-English Notes
- Built a single `DateTime` column from whatever the dataset provided (flexible parsing with comma/semicolon delimiters and both month-first/day-first formats).
- Looked for missing times and uneven gaps using the distribution of time differences.
- Business takeaway: “On average, there’s a reading roughly every 0 days 00:10:00 — i.e., about 10 minute(s) per record.”
