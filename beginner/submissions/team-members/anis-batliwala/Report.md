**Power Consumption vs Environmental Features: Insight Report**

---

For detailed charts and code, refer to [extra-data-analysis.ipynb](extra-data-analysis.ipynb).

---

### 1. Data integrirty check

- Data is consistent: Timestamps were checked by generating a full time range from start to end using the expected interval, then comparing it to the actual data. No major gaps or irregularities showed up — everything looks mostly consistent.
- Sampling frequency: The dataset uses a 10-minute sampling frequency, confirmed by checking the time differences between entries. Most of the intervals consistently show a 10-minute gap, meaning the records are spaced out regularly across the time series.
- Duplicates: No duplicate or inconsistent DateTime entries showed up. A quick check for duplicates returned zero, and the datetime column was converted to the correct format to keep everything consistent before moving forward.

---

### 2. Insight Summary: Weekly + Hourly Power Consumption Patterns

**Weekly Patterns**

- Zones 1 & 2 (Commercial/Institutional Behavior):
  - Peak Usage: Tuesday to Thursday.
  - Drop: Noticeable decline from Friday to Sunday, lowest on Sunday.
  - This pattern strongly aligns with typical workweek schedules.
  - Indicates usage is driven by office/commercial activity—heaviest midweek, lightest on weekends.
- Zone 3 (Residential Behavior):
  - Power consumption is much flatter across the week.
  - Slight uptick on weekends (especially Saturday).
  - Suggests energy demand is tied to constant residential occupancy, unlike Zones 1 & 2.

**Hourly Patterns**

- Zone 1 & Zone 2:
  - Low usage overnight (00:00–06:00).
  - Sharp ramp-up from 07:00 to 13:00, peaking between 13:00–15:00.
  - Evening spike (18:00–20:00) – possibly due to extended work hours or evening operations.
  - Post 21:00 decline suggests end of business activity.
  - Pattern matches work-hour dependent load, with dual peaks: midday and early evening.
- Zone 3:
  - Very low usage early morning (00:00–07:00).
  - Gradual increase from 08:00, with steady rise until 13:00–17:00.
  - Sharp evening peak from 18:00 to 21:00, aligns with residential usage (cooking, lights, entertainment).
  - Power drops after 22:00 – indicative of night-time routine.

**Combined Insight**
| Feature | Zone 1 & 2 | Zone 3 |
| ----------------- | ---------------------------- | ------------------------------------------ |
| **Weekday Trend** | High midweek, low weekends | Stable all week, slight weekend rise |
| **Hourly Trend** | Dual peaks: midday + evening | Clear **evening** peak only |
| **User Type** | Commercial/office-heavy | Predominantly residential |
| **Usage Nature** | Schedule-driven (9–6 jobs) | Home-living driven (morning/evening usage) |

---

### 3. Strong Autocorrelation (Lag-1)

- All three zones show **strong autocorrelation**, with power consumption at any time closely related to the previous time point.

  - **Zone 3** has the strongest autocorrelation (points tightly aligned along the diagonal in lag-1 plot).
  - **Zone 1 and Zone 2** show similar patterns, but with slightly more scatter.

---

### 4. Environmental Correlation Insights

#### Temperature

- **Positively correlated** with power consumption across all zones:

  - Zone 3: 0.49
  - Zone 1: 0.44
  - Zone 2: 0.38

- Key factor for high power consumption, especially during summer months due to air conditioning demand.

#### Humidity

- **Moderate negative correlation** in all zones (around -0.23 to -0.29).
- High humidity generally increases discomfort, but its inverse correlation may be due to seasonal patterns (e.g., high humidity in cooler months).

#### Wind Speed

- Weak to moderate positive correlation:

  - Strongest in Zone 3 (0.28).
  - May influence thermal comfort and building ventilation needs.

#### Diffuse and General Diffuse Flows

- Weak correlation overall:

  - **General Diffuse Flows** slightly higher in Zone 1 (0.18).
  - These flows represent indirect solar radiation and have minor impacts on indoor temperature control.

---

### 5. Inter-Zone Relationships

- **Zone 1 & Zone 2** have very high correlation (\~0.83):

  - Indicates shared usage patterns (likely similar occupant behavior or operational schedules).

- **Zone 3** has strong but lower correlations:

  - With Zone 1: 0.75
  - With Zone 2: 0.57
  - Suggests more distinct behavior, possibly due to different occupancy or function.

---

### Summary Table

| Feature               | Zone 1 | Zone 2 | Zone 3 | Interpretation                             |
| --------------------- | ------ | ------ | ------ | ------------------------------------------ |
| Temperature           | 0.44   | 0.38   | 0.49   | Strongest influence on power consumption   |
| Humidity              | -0.29  | -0.29  | -0.23  | Inversely related, possibly due to seasons |
| Wind Speed            | 0.17   | 0.15   | 0.28   | Slightly increases usage, esp. in Zone 3   |
| Diffuse Flows         | 0.08   | 0.04   | -0.04  | Minor influence, potentially negligible    |
| General Diffuse Flows | 0.18   | 0.15   | 0.06   | Minor effect, stronger in Zone 1           |

---

### Final Insights

- Temperature remains the **dominant environmental factor** affecting power consumption across all zones.
- Humidity and wind show **secondary impacts**, with humidity typically inversely correlated.
- Diffuse solar radiation has **minimal but detectable effect**.
- Power usage patterns are **strongly predictable** from past usage due to high autocorrelation.
- Zones 1 & 2 behave similarly, while **Zone 3 shows more consistent and distinct patterns**.

---
