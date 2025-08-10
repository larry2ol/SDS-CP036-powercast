# 💼 Week 1 – Section1: Time Consistency & Structure (Business-Friendly Report)

## Key Questions Answered
**Q1: Are there any missing or irregular timestamps in the dataset? How did you verify consistency?**  
I converted the date and time into a single timeline and checked the gaps between readings. Most records are evenly spaced; I used the differences between rows to confirm the pattern.

**Q2: What is the sampling frequency and are all records spaced consistently?**  
I measured the time between consecutive rows. The most common spacing is **0 days 00:01:00**, which suggests the intended sampling rate.

**Q3: Did you encounter any duplicates or inconsistent `DateTime` entries?**  
I checked for repeated timestamps. I found **0** duplicates; these could be rechecked or removed if needed.

## Plain-English Notes
- I first made a proper date-time column.
- I looked for missing times and uneven gaps.
- If a business user asked, I’d explain the cadence like this: “On average, I get a reading roughly every 0 days 00:01:00 — so a new data point about once every 1 minute(s).”
