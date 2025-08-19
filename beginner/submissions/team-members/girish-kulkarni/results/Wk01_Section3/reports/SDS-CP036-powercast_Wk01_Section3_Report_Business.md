# 💼 Week 1 – Section3: Drivers & Correlations (Business-Friendly Report)

## Dataset
Using file: **Tetuan City power consumption.csv**  
Period: **2017-01-01 00:00:00 → 2017-12-30 23:50:00**  
Rows: **52,416**

## Key Questions Answered
**Q1: Which factors appear most correlated with total and per-zone consumption?**  
- Total consumption driver: **Temperature (|r|=0.73)**  
- Zone 1 driver: **Temperature (|r|=0.73)**  
- Zone 2 driver: **Temperature (|r|=0.42)**  
- Zone 3 driver: **Temperature (|r|=0.61)**

**Q2: Are relationships linear or do they show thresholds?**  
- The **scatter plots** (daily means) help reveal linear vs. curved patterns and potential thresholds. Patterns tend to be smooth with some spread due to operational effects.

**Q3: Which visualizations helped you uncover these patterns?**  
- **Correlation matrix** of daily means: `plots/section3_correlation_heatmap.png`  
- **Scatter plots** contrasting key drivers vs Total_kW: plots/section3_scatter_temp_total.png, plots/section3_scatter_humidity_total.png

## What we computed
- Canonical **DateTime** and zone aliasing (**Zone 1/2/3 → Sub_metering 1/2/3**).  
- Daily means to stabilize noise.  
- Pearson correlations among consumption (Total + Zones) and environmental features.
