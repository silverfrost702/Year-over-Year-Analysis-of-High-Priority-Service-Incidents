import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title='OIT Incident Intelligence',
    layout='wide',
    initial_sidebar_state='collapsed',
)

# ── Color palette ──────────────────────────────────────────────
BLUE       = '#3498db'
ORANGE     = '#e67e22'
BLUE2      = '#2980b9'
ORANGE2    = '#d35400'
GREEN      = '#2ecc71'
RED        = '#e74c3c'
BG         = '#0f172a'
CARD_BG    = '#1e293b'
BORDER     = '#334155'
TEXT       = '#f1f5f9'
SUBTEXT    = '#94a3b8'

AUTOMATED = r'pagerduty|logic.?app|zabbix|monitoring|integration|automated|switch.*down|is down'

# ── Global Plotly layout ────────────────────────────────────────
def apply_layout(fig, title='', height=420):
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=TEXT, family='Inter, sans-serif'), x=0.01),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT, family='Inter, sans-serif'),
        margin=dict(l=50, r=30, t=55, b=60),
        height=height,
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor=BORDER, font=dict(color=TEXT)),
        xaxis=dict(showgrid=False, linecolor=BORDER, tickcolor=BORDER, color=TEXT),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, color=TEXT),
    )
    return fig

# ── Custom CSS (dark theme) ─────────────────────────────────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*='css'] {{ font-family: 'Inter', sans-serif; }}
    .stApp {{ background-color: {BG}; }}
    .block-container {{ padding: 0 2rem 2rem 2rem; max-width: 1400px; }}

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        background: {CARD_BG}; border-radius: 10px; padding: 4px; gap: 4px;
        border: 1px solid {BORDER};
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {SUBTEXT}; background: transparent;
        border-radius: 8px; padding: 8px 20px; font-weight: 500;
    }}
    .stTabs [aria-selected="true"] {{
        color: {TEXT} !important; background: {BG} !important;
        border-bottom: 2px solid {BLUE} !important;
    }}

    /* KPI cards */
    .kpi-card {{
        background: {CARD_BG}; border: 1px solid {BORDER};
        border-radius: 12px; padding: 18px 20px; text-align: center;
    }}
    .kpi-label {{ color: {SUBTEXT}; font-size: 11px; text-transform: uppercase;
                  letter-spacing: 0.08em; margin-bottom: 6px; }}
    .kpi-value {{ font-size: 28px; font-weight: 700; }}

    /* Section headers */
    .section-header {{ color: {SUBTEXT}; font-size: 11px; text-transform: uppercase;
                       letter-spacing: 0.1em; border-bottom: 1px solid {BORDER};
                       padding-bottom: 6px; margin: 20px 0 12px 0; }}

    /* Header bar */
    .header-bar {{
        background: {CARD_BG}; border: 1px solid {BORDER};
        border-radius: 12px; padding: 20px 28px; margin-bottom: 1.2rem;
        display: flex; justify-content: space-between; align-items: center;
    }}
    .header-title {{ font-size: 22px; font-weight: 700; color: {TEXT}; margin: 0; }}
    .header-sub   {{ font-size: 13px; color: {SUBTEXT}; margin: 2px 0 0 0; }}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown(f"""
<div class="header-bar">
  <div>
    <p class="header-title">OIT Incident Intelligence</p>
    <p class="header-sub">Year-over-Year P1 / P2 Service Incident Analysis</p>
  </div>
  <p style="color:{SUBTEXT};font-size:12px;margin:0;">University IT Operations</p>
</div>
""", unsafe_allow_html=True)


# ── Data loading & caching ─────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('Data/Final_Cleaned_Dataset.csv')
    df['opened at'] = pd.to_datetime(df['opened at'], errors='coerce')
    df = df.dropna(subset=['opened at'])
    df['month_year'] = df['opened at'].dt.to_period('M').astype(str)
    df['month_num']  = df['opened at'].dt.month

    buildings = ['Nedderman Hall', 'Wolf Hall', 'Life Science', 'University Hall',
                 'Pickard Hall', 'ERB', 'COBA', 'Davis Hall']
    issue_map = {
        'Network/Wi-Fi':    ['wifi', 'network', 'switch', 'router', 'connection', 'internet'],
        'Cloud/Server':     ['azure', 'server', 'host', 'vm', 'database', 'zabbix'],
        'Security/Account': ['login', 'account', 'access', 'password', 'compromised', 'mfa'],
    }

    def extract_meta(row):
        text  = str(row['short_description_cleaned']).lower()
        bldg  = next((b for b in buildings if b.lower() in text), 'Campus Wide/Other')
        issue = 'General Technical'
        for cat, terms in issue_map.items():
            if any(t in text for t in terms): issue = cat; break
        return pd.Series([bldg, issue])

    df[['building', 'issue_type']] = df.apply(extract_meta, axis=1)
    return df

# Module-level helper — must be outside cached function for pickling
def to_dt(idx):
    return idx.to_timestamp() if hasattr(idx, 'to_timestamp') else pd.to_datetime(idx)

@st.cache_data
def compute_all(_df):
    df = _df.copy()

    def iqr_clean(s):
        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
        return s.clip(Q1 - 1.5*(Q3-Q1), Q3 + 1.5*(Q3-Q1)).fillna(s.median())

    # Monthly volume
    monthly = df.groupby('month_year').size().reset_index(name='count')

    # Top 5 groups
    top5 = df['assignment_group'].value_counts().head(5).reset_index()
    top5.columns = ['group', 'count']

    # Pareto
    svc = df['u_business_service'].value_counts().reset_index()
    svc.columns = ['service', 'count']
    svc['cum_pct'] = 100 * svc['count'].cumsum() / svc['count'].sum()

    # P1 stacked
    p1_all = df[df['priority'] == 'P1'].copy()
    p1_all['source'] = p1_all['short_description_cleaned'].str.contains(
        AUTOMATED, case=False, na=False, regex=True
    ).map({True: 'Automated', False: 'Human-Reported'})
    p1_stack = p1_all.groupby(['month_year', 'source']).size().unstack(fill_value=0).reset_index()

    # P1 human only
    p1h = p1_all[p1_all['source'] == 'Human-Reported'].copy()

    # Resolvers — before = all P1, after = Logic App rows only removed
    LOGIC_APP = r'logic.?app'
    p1_logic_removed = p1_all[~p1_all['short_description_cleaned'].str.contains(
        LOGIC_APP, case=False, na=False, regex=True)]
    tb = p1_all['assigned_to'].value_counts().head(3).reset_index()
    tb.columns = ['name', 'before']
    ta = p1_logic_removed['assigned_to'].value_counts().head(3).reset_index()
    ta.columns = ['name', 'after']
    resolvers = tb.merge(ta, on='name', how='left').fillna(0)
    total_before_res = len(p1_all)
    total_after_res  = len(p1_logic_removed)
    removed_res      = total_before_res - total_after_res

    # P1 forecast (YoY mean)
    p1_mon = p1h.groupby('month_year').size()
    p1_mm  = p1h.groupby('month_num').size().div(
        p1h.groupby('month_num')['month_year'].nunique()
    ).round().astype(int)
    last_p  = pd.Period(p1_mon.index[-1], freq='M')
    fc_idx  = [last_p + i for i in range(1, 7)]
    fc_vals = [int(p1_mm.get(p.month, int(p1_mon.median()))) for p in fc_idx]
    fc_strs = [str(p) for p in fc_idx]

    # P2 weekly forecast
    p2_raw    = df[df['priority'] == 'P2'].set_index('opened at')
    p2_wk     = iqr_clean(p2_raw.resample('W').size().fillna(0))
    p2_wk_dt  = to_dt(p2_wk.index)
    try:
        hw      = ExponentialSmoothing(p2_wk, trend='add', seasonal='add',
                                       seasonal_periods=52,
                                       initialization_method='estimated').fit()
        p2_fc   = hw.forecast(12)
        p2_fc_dt = to_dt(p2_fc.index)
    except Exception:
        p2_fc, p2_fc_dt = pd.Series(dtype=float), pd.DatetimeIndex([])

    # P1 decomposition
    p1_ms = iqr_clean(p1h.set_index('opened at').resample('ME').size().fillna(0))
    p1_dt = to_dt(p1_ms.index)
    try:    p1_d = seasonal_decompose(p1_ms, model='additive', period=4)
    except: p1_d = None

    # P2 decomposition
    try:    p2_d = seasonal_decompose(p2_wk, model='additive', period=52)
    except: p2_d = None

    return dict(
        monthly=monthly, top5=top5, svc=svc,
        p1_stack=p1_stack, p1_all=p1_all, p1h=p1h,
        resolvers=resolvers, total_before_res=total_before_res,
        total_after_res=total_after_res, removed_res=removed_res,
        p1_mon=p1_mon, fc_vals=fc_vals, fc_strs=fc_strs,
        p2_wk=p2_wk, p2_wk_dt=p2_wk_dt, p2_fc=p2_fc, p2_fc_dt=p2_fc_dt,
        p1_d=p1_d, p1_dt=p1_dt, p2_d=p2_d,
    )

# ── Location & Issue data ──────────────────────────────────────
LOGIC_PATTERNS = ['PAGERDUTY', 'LOGIC APP', 'FAILED LOGIC APP', 'FAILED LOGIC', 'LOGIC FAILURE']

@st.cache_data
def load_location_data():
    df_loc  = pd.read_csv('Data/BldgIDsDataset.csv', encoding='latin1')
    bldg    = pd.read_excel('Data/Building IDs.xlsx')
    RCOL    = 'Campus Region'
    PCOL    = 'priority'
    DCOL    = 'short_description_cleaned'
    for col in [RCOL, PCOL, DCOL]:
        if col in df_loc.columns:
            df_loc[col] = df_loc[col].astype(str).str.strip().str.upper()
    bldg[RCOL] = bldg[RCOL].astype(str).str.strip().str.upper()
    for col in ['opened at', 'closed at']:
        if col in df_loc.columns:
            df_loc[col] = pd.to_datetime(df_loc[col], errors='coerce')
    loc_names = set(bldg[RCOL].dropna().unique())
    return df_loc, loc_names

@st.cache_data
def compute_location(_df_loc, _loc_names):
    df  = _df_loc.copy()
    lns = _loc_names
    RCOL = 'Campus Region'
    PCOL = 'priority'
    DCOL = 'short_description_cleaned'

    location_df = df[df[RCOL].isin(lns)].copy()
    issue_df    = df[~df[RCOL].isin(lns)].copy()

    # ── Chart 1 data: top 10 locations P1/P2
    loc_p12 = location_df[location_df[PCOL].isin(['P1', 'P2'])]
    top_locs = loc_p12[RCOL].value_counts().head(10).index
    plot_loc = (loc_p12[loc_p12[RCOL].isin(top_locs)]
                .groupby([RCOL, PCOL]).size().unstack(fill_value=0))
    for c in ['P1', 'P2']:
        if c not in plot_loc.columns: plot_loc[c] = 0
    plot_loc['Total'] = plot_loc['P1'] + plot_loc['P2']
    plot_loc = plot_loc.sort_values('Total')

    # ── Chart 2 data: issue groups (excl. logic app)
    issue_clean = issue_df[
        (~issue_df[DCOL].str.contains('|'.join(LOGIC_PATTERNS), na=False)) &
        (issue_df[PCOL].isin(['P1', 'P2']))
    ].copy()

    def group_issue(t):
        if any(x in t for x in ['COMPROMISED','HACKED','SECURITY BREACH']): return 'Security / Compromised Accounts'
        if any(x in t for x in ['PHISHING','MALWARE','DEFENDER','THREAT','SPAM']): return 'Security Alerts / Phishing'
        if any(x in t for x in ['NETWORK','WIFI','WI-FI','INTERNET','LATENCY','ETHERNET']): return 'Network Issues'
        if any(x in t for x in ['OUTAGE','DOWN','NOT WORKING','SERVICE DOWN']): return 'System Outages'
        if any(x in t for x in ['LOGIN','SIGN IN','AUTH','PASSWORD','ACCESS','LOCKED OUT','LOCKED']): return 'Login / Access Issues'
        if any(x in t for x in ['EMAIL','OUTLOOK','MAILBOX']): return 'Email Issues'
        if any(x in t for x in ['PHONE','CALL','VOICEMAIL','TEAMS CALL']): return 'Phone / Communication Issues'
        if any(x in t for x in ['PRINTER','PRINT','PHAROS']): return 'Printing Issues'
        if 'VPN' in t: return 'VPN Issues'
        if any(x in t for x in ['SOFTWARE','APPLICATION','APP ERROR','MYMAV','CANVAS','SITECORE','UTSHARE']): return 'Application Issues'
        if any(x in t for x in ['HARDWARE','LAPTOP','DEVICE','COMPUTER','MONITOR','PROJECTOR']): return 'Hardware Issues'
        return 'Other / Uncategorized'

    issue_clean['issue_group'] = issue_clean[DCOL].apply(group_issue)
    plot_issue = issue_clean.groupby(['issue_group', PCOL]).size().unstack(fill_value=0)
    for c in ['P1', 'P2']:
        if c not in plot_issue.columns: plot_issue[c] = 0
    plot_issue['Total'] = plot_issue['P1'] + plot_issue['P2']
    plot_issue = plot_issue.sort_values('Total')

    # ── Tables: incident group summaries (excl. auto)
    df_tbl = df[~df[DCOL].str.contains('|'.join(LOGIC_PATTERNS), na=False)].copy()

    replacements = {
        'M365 DEFENDER ALERT': 'M365 DEFENDER INCIDENT ALERT',
        'DEFENDER INCIDENT ALERT': 'M365 DEFENDER INCIDENT ALERT',
        'PASSWORD RESET': 'PASSWORD / ACCESS ISSUE',
        'ACCOUNT LOCKED': 'PASSWORD / ACCESS ISSUE',
        'LOCKED OUT': 'PASSWORD / ACCESS ISSUE',
        'VPN NOT WORKING': 'VPN ISSUE',
        'OUTLOOK ISSUE': 'EMAIL ISSUE',
    }
    def normalize(text):
        t = str(text).upper().strip()
        for k, v in replacements.items():
            if k in t: return v
        return t

    df_tbl['incident_group'] = df_tbl[DCOL].apply(normalize)

    NCOL = 'number'; ACOL = 'assigned_to'; SDCOL = 'short_description'
    OCOL = 'opened at'; CCOL = 'closed at'

    def build_summary(data, pval, top_n=10):
        temp = data[data[PCOL] == pval]
        if temp.empty: return pd.DataFrame()
        has_sd = SDCOL in data.columns
        agg_args = {
            'Total Occurrences': (NCOL, 'count'),
            'Priority':          (PCOL, lambda x: x.mode().iloc[0] if not x.mode().empty else pval),
            'Primary Assignee':  (ACOL, lambda x: x.mode().iloc[0] if not x.mode().empty else ''),
            'First Opened':      (OCOL, 'min'),
            'Latest Closed':     (CCOL, 'max'),
            'Sample Incident IDs': (NCOL, lambda x: ', '.join(x.astype(str).head(3))),
        }
        if has_sd:
            agg_args['Sample Description'] = (SDCOL, lambda x: x.dropna().iloc[0] if len(x.dropna()) > 0 else '')
        g = (temp.groupby('incident_group').agg(**agg_args)
             .reset_index().rename(columns={'incident_group': 'Incident Group'})
             .sort_values('Total Occurrences', ascending=False).head(top_n))
        for col in ['First Opened', 'Latest Closed']:
            if col in g.columns:
                g[col] = pd.to_datetime(g[col], errors='coerce').dt.strftime('%m-%d-%Y %H:%M').fillna('-')
        g['Primary Assignee'] = g['Primary Assignee'].replace(['nan','None','null'], '-')
        if 'Sample Description' in g.columns:
            g['Sample Description'] = g['Sample Description'].replace(['nan','None','null'], '-')
        return g

    top_p1_tbl = build_summary(df_tbl, 'P1')
    top_p2_tbl = build_summary(df_tbl, 'P2')

    return dict(
        plot_loc=plot_loc, plot_issue=plot_issue,
        top_p1_tbl=top_p1_tbl, top_p2_tbl=top_p2_tbl,
        loc_count=len(location_df), issue_count=len(issue_df),
    )

with st.spinner('Loading data...'):
    df        = load_data()
    data      = compute_all(df)
    try:
        df_loc, loc_names = load_location_data()
        loc_data = compute_location(df_loc, frozenset(loc_names))
        HAS_LOC  = True
    except Exception as _e:
        HAS_LOC  = False
        _loc_err = str(_e)

# ── KPI Bar ────────────────────────────────────────────────────
total   = len(df)
total_p1 = len(df[df['priority'] == 'P1'])
total_p2 = len(df[df['priority'] == 'P2'])
p1h_cnt  = len(data['p1h'])
top_grp  = df['assignment_group'].value_counts().index[0]

k1, k2, k3, k4, k5 = st.columns(5)
for col, label, value, color in [
    (k1, 'Total Incidents',     f'{total:,}',       BLUE),
    (k2, 'P1 Critical',         f'{total_p1:,}',    RED),
    (k3, 'P2 High',             f'{total_p2:,}',    ORANGE),
    (k4, 'P1 Human-Reported',   f'{p1h_cnt}',       GREEN),
    (k5, 'Top Assignment Group', top_grp[:22],       BLUE2),
]:
    col.markdown(f"""
    <div class="kpi-card" style="border-top: 3px solid {color}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value" style="color:{color}">{value}</div>
    </div>""", unsafe_allow_html=True)

st.markdown('<br>', unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    'Overview',
    'Workload Analysis',
    'P1 Deep Dive',
    'P2 Analysis',
    'Decomposition',
    'Location & Issues',
])

# ════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-header">Monthly Incident Volume</p>', unsafe_allow_html=True)
    fig = go.Figure(go.Scatter(
        x=data['monthly']['month_year'], y=data['monthly']['count'],
        mode='lines+markers',
        line=dict(color=BLUE, width=3),
        marker=dict(size=9, color=ORANGE, line=dict(color='white', width=1.5)),
        hovertemplate='%{x}<br>Incidents: %{y}<extra></extra>',
    ))
    apply_layout(fig, 'Monthly Incident Volume Trend', 400)
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">P1 Incidents — Automated vs. Human-Reported</p>',
                unsafe_allow_html=True)
    fig2 = go.Figure()
    ps = data['p1_stack']
    if 'Automated' in ps.columns:
        fig2.add_trace(go.Bar(x=ps['month_year'], y=ps['Automated'],
                              name='Automated (Monitoring)', marker_color=ORANGE,
                              hovertemplate='%{x}<br>Automated: %{y}<extra></extra>'))
    if 'Human-Reported' in ps.columns:
        fig2.add_trace(go.Bar(x=ps['month_year'], y=ps['Human-Reported'],
                              name='Human-Reported', marker_color=BLUE,
                              hovertemplate='%{x}<br>Human: %{y}<extra></extra>'))
    apply_layout(fig2, 'P1 Incidents by Month — Automated vs. Human-Reported', 400)
    fig2.update_layout(barmode='stack')
    fig2.update_xaxes(tickangle=-45)
    st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 2 — WORKLOAD ANALYSIS
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-header">Assignment Group Distribution</p>',
                unsafe_allow_html=True)
    t5 = data['top5']
    colors = [BLUE, BLUE2, '#5dade2', '#85c1e9', '#aed6f1']
    fig3 = go.Figure(go.Bar(
        x=t5['count'], y=t5['group'], orientation='h',
        marker_color=colors,
        text=t5['count'], textposition='outside', textfont=dict(color=TEXT, size=12),
        hovertemplate='%{y}<br>Count: %{x}<extra></extra>',
    ))
    apply_layout(fig3, 'Top 5 Assignment Groups', 400)
    fig3.update_layout(yaxis=dict(autorange='reversed', gridcolor=BORDER,
                                  linecolor=BORDER, color=TEXT))
    fig3.update_xaxes(range=[0, t5['count'].max() * 1.2])
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<p class="section-header">Service Impact — Pareto Principle (80/20 Rule)</p>',
                unsafe_allow_html=True)
    svc = data['svc']
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(x=svc['service'].head(15), y=svc['count'].head(15),
                          name='Incident Count', marker_color=BLUE,
                          hovertemplate='%{x}<br>Count: %{y}<extra></extra>'))
    fig4.add_trace(go.Scatter(x=svc['service'].head(15), y=svc['cum_pct'].head(15),
                              name='Cumulative %', mode='lines+markers',
                              line=dict(color=ORANGE, width=2.5),
                              marker=dict(symbol='diamond', size=8),
                              yaxis='y2',
                              hovertemplate='%{x}<br>Cumulative: %{y:.1f}%<extra></extra>'))
    fig4.add_hline(y=80, line=dict(color=ORANGE2, dash='dash', width=1.5),
                   annotation_text='80% threshold', annotation_font_color=ORANGE2,
                   yref='y2')
    fig4.update_layout(
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT, family='Inter, sans-serif'),
        margin=dict(l=50, r=60, t=55, b=100), height=450,
        title=dict(text='Pareto Analysis: Service Impact (80/20 Rule)',
                   font=dict(size=16, color=TEXT), x=0.01),
        xaxis=dict(tickangle=-45, showgrid=False, linecolor=BORDER, color=TEXT),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER, color=TEXT),
        yaxis2=dict(overlaying='y', side='right', showgrid=False,
                    ticksuffix='%', range=[0, 110], linecolor=BORDER, color=TEXT),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor=BORDER, font=dict(color=TEXT)),
    )
    st.plotly_chart(fig4, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 3 — P1 DEEP DIVE
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-header">Top 3 P1 Resolvers — Before vs. After Logic App Removal</p>',
                unsafe_allow_html=True)
    res = data['resolvers']
    st.caption(
        f"Total P1 Incidents: **{data['total_before_res']}**  |  "
        f"After Logic App Removal: **{data['total_after_res']}**  |  "
        f"Removed: **{data['removed_res']}**"
    )
    fig5 = go.Figure()
    fig5.add_trace(go.Bar(name='Before Logic App Removal', x=res['name'], y=res['before'],
                          marker_color=BLUE, text=res['before'].astype(int),
                          textposition='outside', textfont=dict(color=TEXT),
                          hovertemplate='%{x}<br>Before: %{y}<extra></extra>'))
    fig5.add_trace(go.Bar(name='After Logic App Removal', x=res['name'], y=res['after'],
                          marker_color=ORANGE, text=res['after'].astype(int),
                          textposition='outside', textfont=dict(color=TEXT),
                          hovertemplate='%{x}<br>After: %{y}<extra></extra>'))
    apply_layout(fig5, 'Top 3 P1 Incident Resolvers — Impact of Excluding Logic App Failures', 420)
    fig5.update_layout(
        barmode='group',
        yaxis=dict(range=[0, res[['before','after']].max().max() * 1.3],
                   gridcolor=BORDER, linecolor=BORDER, color=TEXT),
    )
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown('<p class="section-header">P1 Monthly Forecast (Human-Reported Only)</p>',
                unsafe_allow_html=True)
    p1m = data['p1_mon']
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(
        x=list(p1m.index), y=list(p1m.values),
        mode='lines+markers', name='Historical (Human-Reported)',
        line=dict(color=BLUE, width=2.5),
        marker=dict(size=9, color=BLUE, line=dict(color='white', width=1.5)),
        hovertemplate='%{x}<br>P1 Count: %{y}<extra></extra>',
    ))
    fig6.add_trace(go.Scatter(
        x=data['fc_strs'], y=data['fc_vals'],
        mode='lines+markers+text', name='Forecast (YoY Mean)',
        line=dict(color=ORANGE, width=2.5, dash='dash'),
        marker=dict(symbol='diamond', size=10, color=ORANGE,
                    line=dict(color='white', width=1.5)),
        text=[str(v) for v in data['fc_vals']], textposition='top center',
        textfont=dict(color=ORANGE, size=11),
        hovertemplate='%{x}<br>Forecast: %{y}<extra></extra>',
    ))
    fig6.add_vline(x=list(p1m.index)[-1], line=dict(color=SUBTEXT, dash='dot', width=1))
    apply_layout(fig6, 'P1 Monthly Prediction (Human-Reported, Y-axis: 0–10)', 400)
    fig6.update_layout(yaxis=dict(range=[0, 10], dtick=1, gridcolor=BORDER,
                                  linecolor=BORDER, color=TEXT))
    st.plotly_chart(fig6, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 4 — P2 ANALYSIS
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-header">P2 Weekly Forecast (Holt-Winters)</p>',
                unsafe_allow_html=True)
    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(
        x=list(data['p2_wk_dt']), y=list(data['p2_wk'].values),
        mode='lines+markers', name='Historical P2 (Cleaned)',
        line=dict(color=BLUE, width=2),
        marker=dict(size=5, color=BLUE),
        hovertemplate='%{x|%Y-%m-%d}<br>Count: %{y}<extra></extra>',
    ))
    if len(data['p2_fc']) > 0:
        fig7.add_trace(go.Scatter(
            x=list(data['p2_fc_dt']), y=list(data['p2_fc'].values),
            mode='lines+markers', name='Forecast (Holt-Winters)',
            line=dict(color=ORANGE, width=2.5, dash='dash'),
            marker=dict(symbol='diamond', size=8, color=ORANGE),
            hovertemplate='%{x|%Y-%m-%d}<br>Forecast: %{y:.0f}<extra></extra>',
        ))
    apply_layout(fig7, 'P2 Weekly Incident Prediction', 460)
    fig7.update_xaxes(dtick='M2', tickformat='%Y-%m', tickangle=-45)
    st.plotly_chart(fig7, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 5 — DECOMPOSITION
# ════════════════════════════════════════════════════════════
with tab5:
    def decomp_chart(x, y, title, color, mode='lines', h=350):
        fig = go.Figure()
        valid = ~np.isnan(y) if isinstance(y, np.ndarray) else ~y.isna()
        xv = x[valid] if hasattr(x, '__getitem__') else x
        yv = y[valid] if hasattr(y, '__getitem__') else y
        if mode == 'markers':
            fig.add_trace(go.Scatter(x=xv, y=yv, mode='markers',
                                     marker=dict(color=color, size=8),
                                     hovertemplate='%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'))
        else:
            fig.add_trace(go.Scatter(x=xv, y=yv, mode='lines',
                                     line=dict(color=color, width=2.5),
                                     hovertemplate='%{x|%Y-%m}<br>%{y:.2f}<extra></extra>'))
        fig.add_hline(y=0, line=dict(color=SUBTEXT, dash='dot', width=1))
        apply_layout(fig, title, h)
        fig.update_xaxes(dtick='M2', tickformat='%Y-%m', tickangle=-45)
        return fig

    st.markdown('<p class="section-header">P1 Decomposition (Monthly — Human-Reported)</p>',
                unsafe_allow_html=True)
    if data['p1_d']:
        d = data['p1_d']
        dt = data['p1_dt']
        c1, c2 = st.columns(2)
        c1.plotly_chart(decomp_chart(dt, d.trend.values,    'P1 — Trend (Stability)',               BLUE),  use_container_width=True)
        c2.plotly_chart(decomp_chart(dt, d.seasonal.values, 'P1 — Seasonality (Recurring Patterns)', GREEN), use_container_width=True)
        st.plotly_chart(decomp_chart(dt, d.resid.values, 'P1 — Residuals (Unexplained Noise)', RED, 'markers'), use_container_width=True)
    else:
        st.warning('Not enough P1 data for decomposition.')

    st.markdown('<p class="section-header">P2 Decomposition (Weekly)</p>', unsafe_allow_html=True)
    if data['p2_d']:
        d2  = data['p2_d']
        dt2 = to_dt(data['p2_wk'].index)
        c3, c4 = st.columns(2)
        c3.plotly_chart(decomp_chart(dt2, d2.trend.values,    'P2 — Trend (Stability)',               BLUE),  use_container_width=True)
        c4.plotly_chart(decomp_chart(dt2, d2.seasonal.values, 'P2 — Seasonality (Recurring Patterns)', GREEN), use_container_width=True)
        st.plotly_chart(decomp_chart(dt2, d2.resid.values, 'P2 — Residuals (Unexplained Noise)', RED, 'markers'), use_container_width=True)
    else:
        st.warning('Not enough P2 data for decomposition.')

# ════════════════════════════════════════════════════════════
# TAB 6 — LOCATION & ISSUES
# ════════════════════════════════════════════════════════════
with tab6:
    if not HAS_LOC:
        st.error(f'Could not load location dataset: {_loc_err}')
    else:
        def loc_bar(plot_df, title, note=None, h=460):
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=plot_df.index, x=plot_df['P1'], orientation='h', name='P1',
                marker_color=BLUE,
                text=plot_df['P1'].apply(lambda v: str(v) if v > 0 else ''),
                textposition='inside', insidetextanchor='middle',
                hovertemplate='%{y}<br>P1: %{x}<extra></extra>',
            ))
            fig.add_trace(go.Bar(
                y=plot_df.index, x=plot_df['P2'], orientation='h', name='P2',
                marker_color=ORANGE,
                text=plot_df['P2'].apply(lambda v: str(v) if v > 0 else ''),
                textposition='inside', insidetextanchor='middle',
                hovertemplate='%{y}<br>P2: %{x}<extra></extra>',
            ))
            anns = []
            if note:
                anns.append(dict(text=note, xref='paper', yref='paper',
                                 x=0.5, y=-0.12, showarrow=False,
                                 font=dict(size=10, color=SUBTEXT)))
            apply_layout(fig, title, h)
            fig.update_layout(barmode='stack',
                              yaxis=dict(gridcolor=BORDER, linecolor=BORDER, color=TEXT),
                              margin=dict(l=220, r=30, t=55, b=80),
                              annotations=anns)
            return fig

        def make_tbl_fig(df_t, title):
            if df_t.empty:
                return go.Figure()
            col_widths = [220, 70, 90, 260, 120, 120, 120, 180]
            return go.Figure(data=[go.Table(
                columnwidth=col_widths[:len(df_t.columns)],
                header=dict(
                    values=[f'<b>{c}</b>' for c in df_t.columns],
                    fill_color='#1F4E79', font=dict(color='white', size=12),
                    align='left', height=36,
                ),
                cells=dict(
                    values=[df_t[c] for c in df_t.columns],
                    fill_color=[[CARD_BG if i%2==0 else BG for i in range(len(df_t))]
                                for _ in df_t.columns],
                    align='left', font=dict(color=TEXT, size=11), height=32,
                ),
            )]).update_layout(
                paper_bgcolor=CARD_BG,
                title=dict(text=f'<b>{title}</b>', x=0,
                           font=dict(size=15, color=TEXT)),
                margin=dict(l=10, r=10, t=45, b=10), height=420,
            )

        st.caption(
            f"Location-based rows: **{loc_data['loc_count']:,}**  |  "
            f"Issue-based rows: **{loc_data['issue_count']:,}**"
        )

        st.markdown('<p class="section-header">Top 10 Affected Campus Locations (P1 vs P2)</p>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            loc_bar(loc_data['plot_loc'], 'Top 10 Affected Locations by Priority (P1 vs P2)'),
            use_container_width=True,
        )

        st.markdown('<p class="section-header">Issue-Based Incident Distribution (Excl. Logic App & PagerDuty)</p>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            loc_bar(
                loc_data['plot_issue'],
                'Issue-Based Incident Distribution by Problem Type',
                note='Logic App & PagerDuty auto-resolved incidents excluded',
                h=500,
            ),
            use_container_width=True,
        )

        st.markdown('<p class="section-header">Top Incident Groups — Detailed Summary Tables</p>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            make_tbl_fig(loc_data['top_p1_tbl'],
                         'Most Occurring P1 Incident Groups (Excl. PagerDuty & Logic App Failures)'),
            use_container_width=True,
        )
        st.plotly_chart(
            make_tbl_fig(loc_data['top_p2_tbl'],
                         'Most Occurring P2 Incident Groups (Excl. PagerDuty & Logic App Failures)'),
            use_container_width=True,
        )
