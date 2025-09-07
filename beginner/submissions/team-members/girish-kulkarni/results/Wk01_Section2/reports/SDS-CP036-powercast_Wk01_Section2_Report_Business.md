# 💼 Week 1 – Section2: Temporal Trends (Business-Friendly Report)

## Dataset
Using file: **Tetuan City power consumption.csv**  
Period: **2017-01-01 00:00:00 → 2017-12-30 23:50:00**  
Rows: **52,416**

## Key Questions Answered
**Q1: What daily or weekly patterns are observable in power consumption across the three zones?**  
- The **line plot** of daily averages (Total & Zones) shows overall movement and relative contribution by zone.  
- By weekday, average total usage peaks on **Thu** based on the dataset's mean profile.

**Q2: Are there seasonal or time-of-day peaks and dips in energy usage?**  
- The **heatmap** for Zone 1 (kitchen proxy) highlights typical time-of-day peaks; the highest average hour is around **20:00**.  
- Broader seasonal effects can be explored by comparing monthly averages (extendable in this section if needed).

**Q3: Which visualizations helped you uncover these patterns?**  
- **Line plot** (daily averages): `plots/section2_daily_averages.png`  
- **Box plot** (by day of week): `plots/section2_box_by_dow.png`  
- **Heatmap** (hour vs day for Zone 1): `plots/section2_heatmap_zone1.png`

## What we computed
- Canonical **DateTime** and zone aliasing (**Zone 1/2/3 → Sub_metering_1/2/3**).  
- **Daily averages** of total and per-zone consumption.  
- **Day-of-week distribution** (box plot) for total consumption.  
- **Hour × day heatmap** for Zone 1.

