<div align="center">

# OIT Incident Intelligence
**Year-over-Year P1 & P2 Service Incident Analysis**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)

*An advanced Data Science & Analytics dashboard predicting workload demands and categorizing campus-wide high-priority service incidents.*

</div>

---

## Project Overview

The **OIT Incident Intelligence** platform is an interactive, analytical dashboard tailored for University IT Operations. It provides deep, data-driven insights into historical **P1 (Critical)** and **P2 (High Priority)** service incidents, performing time-series forecasting, geographical hotspot tracking, and natural language categorization to optimize operational efficiency.

### Core Capabilities
- **Time-Series Forecasting**: Utilizing *Holt-Winters Exponential Smoothing* and YoY median calculations to forecast future P1 and P2 incident volumes.
- **Signal-to-Noise Reduction**: Automatically parsing out monitoring noise (like Azure Logic App or PagerDuty integrations) from human-reported, actionable incidents.
- **Geographic & Issue Analytics**: Heat-mapping the top physical buildings with network/hardware issues versus broad IT/Cloud service failures.
- **Decomposition**: Splitting incident data into *Trend*, *Seasonality*, and *Residual Noise* components.

---

## Visualizing the Trends

### P1 (Critical) Incident Prediction
P1 incidents demand immediate attention. By filtering out automated system anomalies and focusing solely on human-reported operations, we establish a clean baseline for predicting future operational spikes.

![P1 Forecast](Visualizations/p1_forecast.png)

### P1 (Critical) Seasonality
Understanding our data composition means evaluating **Trend** (overall increase or decrease), **Seasonality** (cyclical semester demands), and **Residuals** (unpredictable outliers).

![P1 Seasonality](Visualizations/p1_seasonality.png)

### P2 (High Priority) Forecasting
Using Holt-Winters algorithms to predict the P2 trajectory seamlessly over dynamic semester operations and network load variations.

![P2 Forecast](Visualizations/p2_forecast.png)

---

## Tech Stack & Methods
* **Dashboard Engine**: `Streamlit` with custom CSS mapping and Plotly dynamic visual integration.
* **Data Processing**: `Pandas` and `Numpy` for IQR-based outlier correction, data cleansing, and period re-indexing.
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
