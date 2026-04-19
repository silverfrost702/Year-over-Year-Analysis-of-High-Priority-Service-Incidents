<div align="center">

# OIT Incident Intelligence
**Year-over-Year P1 & P2 Service Incident Analysis**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)

*An advanced Data Science & Analytics dashboard extracting actionable business insights and predicting workload demands from campus-wide service incidents.*
*View Live: https://yoy-p1p2-incidents.streamlit.app/*

</div>

---

## Actionable Business Insights

By scraping and processing thousands of **ServiceNow ITSM tickets**, we translated raw logs into data-driven strategic decisions for OIT management:

### 1. Signal-to-Noise Reduction 
**Insight:** A vast majority of "Critical" incidents were automated noise (e.g., Azure Logic Apps, PagerDuty), not actual outages.  
**Impact:** Filtering these out prevented the Help Desk from chasing false alarms, recovering engineering hours for real campus-wide issues.

### 2. Service Impact (80/20 Rule)
**Insight:** Analyzing incident frequencies revealed that **20% of core IT services caused 80% of the high-priority workload**.  
**Impact:** OIT management can prioritize preventative maintenance budgets on this critical subset, reducing future incident volume at its root.

### 3. Geographical & Issue Hotspots
**Insight:** We manually mapped incident logs to categories ("Network", "Cloud", "Security") for maximum precision, cross-referencing them to the top 10 physical campus buildings.  
**Impact:** Highlighted structural bottlenecks. OIT can now easily visualize which specific buildings suffer physical network degradation versus cloud login failures.

---

## Predictive Modeling & Trend Analysis

We implemented predictive modeling utilizing **Holt-Winters Exponential Smoothing** and **Seasonal Decomposition** to anticipate future workload demands.

### P1 (Critical) Prediction & Seasonality
By focusing solely on human-reported operations, we established a clean baseline for predicting future spikes and evaluating **Seasonality** (recurring semester demands).

![P1 Forecast](Visualizations/p1_forecast.png)
![P1 Seasonality](Visualizations/p1_seasonality.png)

### P2 (High Priority) Forecasting
We smoothed out irregular peaks in P2 volume to seamlessly predict trajectory across dynamic semester operations.

![P2 Forecast](Visualizations/p2_forecast.png)

---

## Tech Stack & Methods
* **Dashboard Engine**: `Streamlit` with custom CSS mapping and Plotly dynamic visual integration.
* **Data Processing Pipeline**: `Pandas` and `Numpy` for IQR-based outlier correction, data cleansing, text scraping, and period re-indexing.
* **Predictive Modeling**: `Statsmodels` (Holt-Winters Exponential Smoothing and Seasonal Decomposition).
* **Data Visualization**: `Matplotlib` for static diagnostic charts and `Plotly` for interactive web dashboards.

---

## Running Locally

Want to explore the dashboard locally? Run the Streamlit application directly:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the Streamlit dashboard
streamlit run app.py
```

<div align="center">
<br/>
<sub>Built by <a href="https://github.com/silverfrost702" target="_blank">silverfrost702</a></sub>
</div>
