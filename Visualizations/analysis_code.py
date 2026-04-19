
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ── Color Palette ──────────────────────────────────────────────
BLUE_PRIMARY     = '#3498db'
ORANGE_PRIMARY   = '#e67e22'
BLUE_SECONDARY   = '#2980b9'
ORANGE_SECONDARY = '#d35400'

AUTOMATED = r'pagerduty|logic.?app|zabbix|monitoring|integration|automated|switch.*down|is down'

# --- 1. DATA PREPARATION ---
df = pd.read_csv('data/Final_Cleaned_Dataset.csv')
df['opened at'] = pd.to_datetime(df['opened at'], errors='coerce')
df = df.dropna(subset=['opened at'])

buildings = ['Nedderman Hall', 'Wolf Hall', 'Life Science', 'University Hall',
             'Pickard Hall', 'ERB', 'COBA', 'Davis Hall']
issue_map = {
    'Network/Wi-Fi': ['wifi', 'network', 'switch', 'router', 'connection', 'internet'],
    'Cloud/Server':  ['azure', 'server', 'host', 'vm', 'database', 'zabbix'],
    'Security/Account': ['login', 'account', 'access', 'password', 'compromised', 'mfa']
}

def extract_metadata(row):
    text  = str(row['short_description_cleaned']).lower()
    bldg  = next((b for b in buildings if b.lower() in text), 'Campus Wide/Other')
    issue = 'General Technical'
    for cat, terms in issue_map.items():
        if any(t in text for t in terms): issue = cat; break
    return pd.Series([bldg, issue])

df[['building', 'issue_type']] = df.apply(extract_metadata, axis=1)


# --- 2. OUTLIER REMOVAL ---
def remove_outliers_iqr(series, label=''):
    Q1, Q3  = series.quantile(0.25), series.quantile(0.75)
    IQR     = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    median  = series.median()
    outliers = series[(series < lower) | (series > upper)]
    if not outliers.empty:
        print(f"[{label}] Outliers capped at median ({median:.0f}):")
        for idx, val in outliers.items():
            print(f"   • {idx} → {val:.0f}  →  {median:.0f}")
    return series.clip(lower=lower, upper=upper).fillna(median)


# --- 3. DECOMPOSITION: 3 SEPARATE CHARTS ---
def plot_decomposition(decomp, p_level, freq_label):
    bg = '#f8f9fa'

    # Convert PeriodIndex → DatetimeIndex for clean axis formatting
    def to_dt(idx):
        if hasattr(idx, 'to_timestamp'):
            return idx.to_timestamp()
        return pd.to_datetime(idx)

    trend_idx    = to_dt(decomp.trend.index)
    seasonal_idx = to_dt(decomp.seasonal.index)
    resid_idx    = to_dt(decomp.resid.index)

    locator   = mdates.MonthLocator(interval=2)
    formatter = mdates.DateFormatter('%Y-%m')

    # Trend
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    ax.plot(trend_idx, decomp.trend.values, color=BLUE_PRIMARY, linewidth=2.5)
    ax.xaxis.set_major_locator(locator); ax.xaxis.set_major_formatter(formatter)
    ax.set_title(f'{p_level} {freq_label} — Trend (Stability)', fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel('Incident Count', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{p_level.lower()}_trend.png', bbox_inches='tight')
    plt.show()

    # Seasonality
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    ax.plot(seasonal_idx, decomp.seasonal.values, color='#27ae60', linewidth=2)
    ax.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.5)
    ax.xaxis.set_major_locator(locator); ax.xaxis.set_major_formatter(formatter)
    ax.set_title(f'{p_level} {freq_label} — Seasonality (Recurring Patterns)', fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel('Seasonal Effect', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{p_level.lower()}_seasonality.png', bbox_inches='tight')
    plt.show()

    # Residuals
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    ax.plot(resid_idx, decomp.resid.values, '.', color='#c0392b', markersize=8)
    ax.axhline(0, color='grey', linestyle='--', linewidth=1, alpha=0.5)
    ax.xaxis.set_major_locator(locator); ax.xaxis.set_major_formatter(formatter)
    ax.set_title(f'{p_level} {freq_label} — Residuals (Unexplained Noise)', fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel('Residual', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{p_level.lower()}_residuals.png', bbox_inches='tight')
    plt.show()


# --- 4. MAIN ANALYSIS FUNCTION ---
def analyze_priority(p_level):
    print(f"\n{'='*50}\nAnalyzing {p_level}\n{'='*50}")

    if p_level == 'P1':
        subset = df[
            (df['priority'] == 'P1') &
            (~df['short_description_cleaned'].str.contains(AUTOMATED, case=False, na=False, regex=True))
        ]
        print(f"P1 human-reported incidents: {len(subset):,}")
    else:
        subset = df[df['priority'] == p_level]

    data_subset   = subset.set_index('opened at')
    weekly_series = data_subset.resample('W').size().fillna(0)

    if len(weekly_series) >= 104:
        period, raw_series, freq_label = 52, weekly_series, "Weekly"
    else:
        raw_series  = data_subset.resample('M').size().fillna(0)
        period, freq_label = 4, "Monthly"
        print(f"Note: {len(weekly_series)} weeks — using Monthly analysis.")

    model_series = remove_outliers_iqr(raw_series, label=p_level)

    # Decomposition
    decomp = seasonal_decompose(model_series, model='additive', period=period)
    plot_decomposition(decomp, p_level, freq_label)

    # ── Convert index to DatetimeIndex for clean x-axis ──
    if hasattr(model_series.index, 'to_timestamp'):
        hist_dt = model_series.index.to_timestamp()
    else:
        hist_dt = pd.to_datetime(model_series.index)

    # Forecast
    if p_level == 'P1':
        s2 = subset.copy()
        s2['month_num']  = s2['opened at'].dt.month
        s2['month_year'] = s2['opened at'].dt.to_period('M').astype(str)
        month_mean = s2.groupby('month_num').size().div(
            s2.groupby('month_num')['month_year'].nunique()).round().astype(int)

        last_period    = pd.Period(model_series.index[-1].strftime('%Y-%m'), freq='M')
        future_periods = [last_period + i for i in range(1, 7)]
        forecast_vals  = [int(month_mean.get(p.month, int(model_series.median()))) for p in future_periods]
        forecast_dt    = pd.to_datetime([str(p) for p in future_periods])
        forecast_label = 'Forecast (YoY Monthly Mean)'
    else:
        hw = ExponentialSmoothing(
            model_series, trend='add', seasonal='add',
            seasonal_periods=period, initialization_method="estimated"
        ).fit()
        fc_raw         = hw.forecast(6 if freq_label == "Monthly" else 12)
        forecast_vals  = fc_raw.values.tolist()
        if hasattr(fc_raw.index, 'to_timestamp'):
            forecast_dt = fc_raw.index.to_timestamp()
        else:
            forecast_dt = pd.to_datetime(fc_raw.index)
        forecast_label = 'Forecast (Holt-Winters)'

    # Forecast chart with clean x-axis
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor('#f8f9fa'); ax.set_facecolor('#f8f9fa')

    ax.plot(hist_dt, model_series.values,
            marker='o', color=BLUE_PRIMARY, linewidth=2.5, markersize=6,
            markeredgecolor='white', markeredgewidth=1.2, label='Historical (Cleaned)')
    ax.plot(forecast_dt, forecast_vals,
            marker='D', color=ORANGE_PRIMARY, linewidth=2.5,
            linestyle='--', markersize=8, markeredgecolor='white', markeredgewidth=1.2,
            label=forecast_label)

    # Clean x-axis: monthly ticks every 2 months
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    if p_level == 'P1':
        ax.set_ylim(0, 10)
        ax.set_yticks(range(0, 11))
        for x, y in zip(forecast_dt, forecast_vals):
            ax.annotate(str(y), (x, y), textcoords='offset points',
                        xytext=(0, 10), ha='center', fontsize=10,
                        fontweight='bold', color=ORANGE_SECONDARY)

    ax.set_title(f'{p_level} {freq_label} Prediction', fontsize=15, fontweight='bold', pad=15)
    ax.set_ylabel('Incident Count', fontsize=12)
    ax.set_xlabel('Timeline', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{p_level.lower()}_forecast.png', bbox_inches='tight')
    plt.show()

    return model_series


# --- 5. EXECUTE ---
p1_data = analyze_priority('P1')
p2_data = analyze_priority('P2')

# --- 6. HOTSPOT MATRIX ---
p1_human = df[
    (df['priority'] == 'P1') &
    (~df['short_description_cleaned'].str.contains(AUTOMATED, case=False, na=False, regex=True))
]
p1_hotspots = pd.crosstab(p1_human['building'], p1_human['issue_type'])
print("\n" + "="*50 + "\nGEOGRAPHIC HOTSPOT AUDIT (P1 — Human-Reported)\n" + "="*50)
print(p1_hotspots.drop('Campus Wide/Other', errors='ignore'))
