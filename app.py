# -*- coding: utf-8 -*-
"""
Portfolio VaR Analysis Dashboard
Multi-Asset Indian Equity Portfolio — Value-at-Risk Analysis
Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import norm, t, skew, kurtosis

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio VaR Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Hero header */
.hero-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    color: white;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -30%;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(108, 99, 255, 0.15), transparent 70%);
    border-radius: 50%;
}
.hero-header h1 {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0 0 0.3rem 0;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    position: relative;
}
.hero-header p {
    font-size: 1.05rem;
    color: rgba(255,255,255,0.7);
    margin: 0;
    position: relative;
}

/* KPI Cards */
.kpi-container {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}
.kpi-card {
    background: linear-gradient(145deg, #1a1a2e, #16213e);
    border: 1px solid rgba(108, 99, 255, 0.2);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    flex: 1;
    min-width: 170px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(108, 99, 255, 0.15);
}
.kpi-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: rgba(255,255,255,0.5);
    margin-bottom: 0.35rem;
}
.kpi-value {
    font-size: 1.55rem;
    font-weight: 700;
    color: #a78bfa;
}
.kpi-value.green { color: #34d399; }
.kpi-value.blue  { color: #60a5fa; }
.kpi-value.amber { color: #fbbf24; }
.kpi-value.rose  { color: #fb7185; }

/* Section headers */
.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #e0e0ff;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(108, 99, 255, 0.3);
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

/* Tables */
.styled-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 2px 15px rgba(0,0,0,0.2);
    margin: 1rem 0;
}
.styled-table thead th {
    background: linear-gradient(135deg, #302b63, #24243e);
    color: #a78bfa;
    padding: 0.9rem 1.2rem;
    text-align: left;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.styled-table tbody td {
    padding: 0.75rem 1.2rem;
    font-size: 0.92rem;
    color: #e0e0e0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.styled-table tbody tr {
    background: #1a1a2e;
    transition: background 0.2s ease;
}
.styled-table tbody tr:hover {
    background: rgba(108, 99, 255, 0.08);
}
.styled-table tbody tr:nth-child(even) {
    background: #16213e;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29, #1a1a2e);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #a78bfa;
}

/* Info badge */
.info-badge {
    display: inline-block;
    background: rgba(108, 99, 255, 0.15);
    border: 1px solid rgba(108, 99, 255, 0.3);
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-size: 0.85rem;
    color: #c4b5fd;
    margin: 0.3rem 0;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Data_FM_project.csv", encoding="latin1")
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.set_index('Date')
    return df


RETURN_COLS = [
    'Rel_Return', 'TCS_Return', 'HDFC_Return', 'ARTL_Return ',
    'INFY_Return', 'ITC_Return', 'Sun_Return', 'ONGC_Return',
    'TATA_return', 'DMART_Return',
]

STOCK_NAMES = [
    'Reliance', 'TCS', 'HDFC Bank', 'Bharti Airtel',
    'Infosys', 'ITC', 'Sun Pharma', 'ONGC',
    'Tata Motors', 'DMart',
]

PRICE_COLS = [
    'RELIANCE', 'TCS', 'HDFCBANK', 'BHARTIARTL',
    'INFY', 'ITC', 'SUNPHARMA', 'ONGC',
    'TATAMOTORS', 'DMART',
]

COLOR_PALETTE = [
    '#6c63ff', '#34d399', '#fb7185', '#fbbf24', '#60a5fa',
    '#a78bfa', '#f472b6', '#38bdf8', '#4ade80', '#facc15',
]

PLOTLY_TEMPLATE = "plotly_dark"


@st.cache_data
def get_returns(df):
    available = [c for c in RETURN_COLS if c in df.columns]
    df_ret = df[available].copy()
    df_ret = df_ret.apply(pd.to_numeric, errors='coerce')
    df_ret = df_ret.dropna()
    return df_ret, available


# ──────────────────────────────────────────────────────────────
# VaR COMPUTATION ENGINE
# ──────────────────────────────────────────────────────────────
def compute_var(port_returns, confidence, ewma_lambda, t_df, portfolio_value):
    """Compute all 5 VaR models. Return a dict of results."""
    percentile = (1 - confidence) * 100
    port_mean = port_returns.mean()
    port_std = port_returns.std()
    z = norm.ppf(1 - confidence)

    # 1) Historical VaR
    hist_var_pct = np.percentile(port_returns, percentile)

    # 2) Normal (Parametric) VaR
    normal_var_pct = port_mean + z * port_std

    # 3) Modified VaR (Cornish-Fisher)
    S = skew(port_returns)
    K = kurtosis(port_returns)
    z_cf = (z
            + (1/6)  * (z**2 - 1) * S
            + (1/24) * (z**3 - 3*z) * K
            - (1/36) * (2*z**3 - 5*z) * (S**2))
    modified_var_pct = port_mean + z_cf * port_std

    # 4) EWMA VaR
    ewma_var_sq = 0.0
    for r in port_returns:
        ewma_var_sq = ewma_lambda * ewma_var_sq + (1 - ewma_lambda) * (r ** 2)
    ewma_vol = np.sqrt(ewma_var_sq)
    ewma_var_pct = z * ewma_vol

    # 5) Student-t VaR
    t_score = t.ppf(1 - confidence, df=t_df)
    t_var_pct = port_mean + t_score * port_std

    models = {
        'Historical': {
            'var_pct': abs(hist_var_pct) * 100,
            'var_val': abs(hist_var_pct) * portfolio_value,
            'color': '#6c63ff',
            'desc': f'Empirical {percentile:.0f}th percentile of returns',
        },
        'Normal': {
            'var_pct': abs(normal_var_pct) * 100,
            'var_val': abs(normal_var_pct) * portfolio_value,
            'color': '#34d399',
            'desc': 'VaR = μ + z·σ (Gaussian assumption)',
        },
        'Modified (CF)': {
            'var_pct': abs(modified_var_pct) * 100,
            'var_val': abs(modified_var_pct) * portfolio_value,
            'color': '#fb7185',
            'desc': 'Cornish-Fisher expansion adjusts for skew & kurtosis',
        },
        'EWMA': {
            'var_pct': abs(ewma_var_pct) * 100,
            'var_val': abs(ewma_var_pct) * portfolio_value,
            'color': '#fbbf24',
            'desc': f'Exponentially Weighted (λ={ewma_lambda})',
        },
        'Student-t': {
            'var_pct': abs(t_var_pct) * 100,
            'var_val': abs(t_var_pct) * portfolio_value,
            'color': '#60a5fa',
            'desc': f't-distribution (df={t_df}) for fat tails',
        },
    }

    stats = {
        'mean': port_mean,
        'std': port_std,
        'skewness': S,
        'kurtosis': K,
        'z_score': z,
        'z_cf': z_cf,
        'raw': {
            'hist': hist_var_pct,
            'normal': normal_var_pct,
            'modified': modified_var_pct,
            'ewma': ewma_var_pct,
            'student_t': t_var_pct,
        }
    }

    return models, stats


def run_monte_carlo(df_returns, available_cols, confidence, n_sim, seed=42):
    """Monte Carlo simulation with random weights."""
    np.random.seed(seed)
    N = len(available_cols)
    mc_vars = np.zeros(n_sim)
    mc_weights_all = np.zeros((n_sim, N))
    percentile = (1 - confidence) * 100

    for i in range(n_sim):
        w = np.random.random(N)
        w = w / w.sum()
        r = df_returns.values @ w
        mc_vars[i] = abs(np.percentile(r, percentile)) * 100
        mc_weights_all[i] = w

    min_idx = np.argmin(mc_vars)
    return mc_vars, mc_weights_all, min_idx


def crisis_var(df_ret, weights, start, end, confidence=0.95):
    """Compute Historical & Parametric VaR for a sub-period."""
    mask = (df_ret.index >= start) & (df_ret.index <= end)
    sub = df_ret[mask]
    if len(sub) < 20:
        return None, None, len(sub)
    r = sub.values @ weights
    h_var = abs(np.percentile(r, (1 - confidence) * 100)) * 100
    p_var = abs(r.mean() + norm.ppf(1 - confidence) * r.std()) * 100
    return h_var, p_var, len(sub)


# ──────────────────────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────────────────────
def main():
    # Load data
    df = load_data()
    df_returns, available_cols = get_returns(df)
    N = len(available_cols)

    # ── SIDEBAR ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Dashboard Controls")
        st.markdown("---")

        portfolio_value = st.slider(
            "💰 Portfolio Value (₹)",
            min_value=100_000, max_value=10_000_000,
            value=1_000_000, step=100_000,
            format="₹%d"
        )

        confidence = st.select_slider(
            "🎯 Confidence Level",
            options=[0.90, 0.95, 0.99],
            value=0.95,
            format_func=lambda x: f"{x:.0%}"
        )

        st.markdown("---")
        st.markdown("### 🔧 Model Parameters")

        ewma_lambda = st.slider(
            "EWMA λ (decay factor)",
            min_value=0.90, max_value=0.99,
            value=0.94, step=0.01,
        )

        t_df = st.slider(
            "Student-t degrees of freedom",
            min_value=3, max_value=30, value=5,
        )

        n_sim = st.slider(
            "Monte Carlo simulations",
            min_value=1_000, max_value=50_000,
            value=10_000, step=1_000,
        )

        st.markdown("---")
        st.markdown(
            "<div class='info-badge'>📅 Data: "
            f"{df_returns.index.min().strftime('%b %Y')} → "
            f"{df_returns.index.max().strftime('%b %Y')}"
            "</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div class='info-badge'>📈 {N} Stocks · "
            f"{len(df_returns)} Trading Days</div>",
            unsafe_allow_html=True
        )

    # ── COMPUTATIONS ─────────────────────────────────────────
    weights = np.array([1 / N] * N)
    port_returns = df_returns.values @ weights

    models, stats = compute_var(
        port_returns, confidence, ewma_lambda, t_df, portfolio_value
    )

    # ── HERO HEADER ──────────────────────────────────────────
    st.markdown("""
    <div class="hero-header">
        <h1>📊 Portfolio VaR Analysis Dashboard</h1>
        <p>Multi-Asset Indian Equity Portfolio — Value-at-Risk Analysis &nbsp;|&nbsp;
        Equal-Weighted · 10 Stocks · """ + f"{confidence:.0%}" + """ Confidence</p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI CARDS ────────────────────────────────────────────
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-label">Portfolio Value</div>
            <div class="kpi-value">₹{portfolio_value:,.0f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Mean Daily Return</div>
            <div class="kpi-value green">{stats['mean']*100:.4f}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Daily Volatility (σ)</div>
            <div class="kpi-value blue">{stats['std']*100:.4f}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Skewness</div>
            <div class="kpi-value amber">{stats['skewness']:.4f}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Excess Kurtosis</div>
            <div class="kpi-value rose">{stats['kurtosis']:.4f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # SECTION 1 — VaR MODEL COMPARISON
    # ══════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">📊 VaR Model Comparison</div>',
                unsafe_allow_html=True)

    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        fig_bar = go.Figure()
        names = list(models.keys())
        pcts = [models[m]['var_pct'] for m in names]
        colors = [models[m]['color'] for m in names]

        fig_bar.add_trace(go.Bar(
            x=names, y=pcts,
            marker_color=colors,
            marker_line_width=0,
            text=[f"{v:.4f}%" for v in pcts],
            textposition='outside',
            textfont=dict(size=13, color='white', family='Inter'),
            hovertemplate='%{x}<br>VaR: %{y:.4f}%<extra></extra>',
        ))
        fig_bar.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=420,
            margin=dict(l=40, r=20, t=50, b=40),
            title=dict(
                text=f"VaR Comparison ({confidence:.0%} Confidence)",
                font=dict(size=16, color='#e0e0ff'),
            ),
            yaxis=dict(
                title="VaR (%)",
                gridcolor='rgba(255,255,255,0.06)',
                zeroline=False,
            ),
            xaxis=dict(tickfont=dict(size=12)),
            bargap=0.3,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_table:
        table_html = """
        <table class="styled-table">
            <thead><tr>
                <th>Model</th><th>VaR %</th><th>₹ Value</th>
            </tr></thead><tbody>
        """
        for name in names:
            m = models[name]
            table_html += f"""
            <tr>
                <td><span style="color:{m['color']};font-weight:600;">●</span> {name}</td>
                <td style="font-weight:600;">{m['var_pct']:.4f}%</td>
                <td>₹{m['var_val']:,.2f}</td>
            </tr>"""
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)

        st.markdown("##### Model Descriptions")
        for name in names:
            st.markdown(
                f"<span style='color:{models[name]['color']};font-weight:600;'>●</span> "
                f"**{name}** — {models[name]['desc']}",
                unsafe_allow_html=True
            )

    # ══════════════════════════════════════════════════════════
    # SECTION 2 — RETURN DISTRIBUTION
    # ══════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">📈 Portfolio Return Distribution</div>',
                unsafe_allow_html=True)

    fig_dist = go.Figure()

    # Histogram
    fig_dist.add_trace(go.Histogram(
        x=port_returns * 100,
        nbinsx=80,
        marker_color='rgba(108, 99, 255, 0.55)',
        marker_line=dict(width=0.5, color='rgba(108,99,255,0.8)'),
        name='Daily Returns',
        histnorm='probability density',
    ))

    # Normal PDF overlay
    x_fit = np.linspace(port_returns.min() * 100, port_returns.max() * 100, 300)
    y_fit = norm.pdf(x_fit, stats['mean'] * 100, stats['std'] * 100)
    fig_dist.add_trace(go.Scatter(
        x=x_fit, y=y_fit,
        mode='lines',
        line=dict(color='rgba(255,255,255,0.45)', width=2, dash='dot'),
        name='Normal Fit',
    ))

    # VaR threshold lines
    raw = stats['raw']
    var_lines = [
        (raw['hist'] * 100, 'Historical', '#6c63ff'),
        (raw['normal'] * 100, 'Normal', '#34d399'),
        (raw['modified'] * 100, 'Modified', '#fb7185'),
        (raw['ewma'] * 100, 'EWMA', '#fbbf24'),
        (raw['student_t'] * 100, 'Student-t', '#60a5fa'),
    ]
    for xv, lbl, col in var_lines:
        fig_dist.add_vline(
            x=xv, line_width=2, line_dash="dash", line_color=col,
            annotation_text=f"{lbl}: {abs(xv):.2f}%",
            annotation_position="top",
            annotation_font=dict(size=10, color=col),
        )

    fig_dist.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=450,
        margin=dict(l=40, r=20, t=60, b=40),
        title=dict(
            text="Daily Return Distribution + VaR Thresholds",
            font=dict(size=16, color='#e0e0ff'),
        ),
        xaxis=dict(title="Daily Return (%)", gridcolor='rgba(255,255,255,0.06)'),
        yaxis=dict(title="Density", gridcolor='rgba(255,255,255,0.06)'),
        legend=dict(
            bgcolor='rgba(0,0,0,0.3)',
            bordercolor='rgba(108,99,255,0.3)',
            borderwidth=1,
            font=dict(size=11),
        ),
        showlegend=True,
        bargap=0.02,
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    # SECTION 3 — CRISIS PERIOD ANALYSIS
    # ══════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">🔥 Crisis Period Analysis</div>',
                unsafe_allow_html=True)

    crisis_periods = {
        '2018 Market Stress': ('2018-01-01', '2018-12-31'),
        'COVID-19 Crash': ('2020-03-01', '2020-12-31'),
        '2022 Inflation': ('2022-01-01', '2022-12-31'),
    }

    crisis_results = {}
    for label, (s, e) in crisis_periods.items():
        h, p, n = crisis_var(df_returns, weights, s, e, confidence)
        if h is not None:
            crisis_results[label] = {'hist': h, 'param': p, 'days': n}

    if crisis_results:
        col_crisis_chart, col_crisis_detail = st.columns([3, 2])

        with col_crisis_chart:
            fig_crisis = go.Figure()
            labels = list(crisis_results.keys())
            h_vals = [crisis_results[l]['hist'] for l in labels]
            p_vals = [crisis_results[l]['param'] for l in labels]

            fig_crisis.add_trace(go.Bar(
                x=labels, y=h_vals,
                name='Historical VaR',
                marker_color='#6c63ff',
                text=[f"{v:.3f}%" for v in h_vals],
                textposition='outside',
                textfont=dict(size=12, color='#c4b5fd'),
            ))
            fig_crisis.add_trace(go.Bar(
                x=labels, y=p_vals,
                name='Parametric VaR',
                marker_color='#fb7185',
                text=[f"{v:.3f}%" for v in p_vals],
                textposition='outside',
                textfont=dict(size=12, color='#fda4af'),
            ))
            fig_crisis.update_layout(
                template=PLOTLY_TEMPLATE,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=420,
                margin=dict(l=40, r=20, t=50, b=40),
                barmode='group',
                bargap=0.25,
                title=dict(
                    text=f"Crisis Period VaR ({confidence:.0%} Confidence)",
                    font=dict(size=16, color='#e0e0ff'),
                ),
                yaxis=dict(
                    title="VaR (%)",
                    gridcolor='rgba(255,255,255,0.06)',
                ),
                legend=dict(
                    bgcolor='rgba(0,0,0,0.3)',
                    bordercolor='rgba(108,99,255,0.3)',
                    borderwidth=1,
                ),
            )
            st.plotly_chart(fig_crisis, use_container_width=True)

        with col_crisis_detail:
            st.markdown("##### Crisis Period Details")
            crisis_table = """
            <table class="styled-table">
                <thead><tr>
                    <th>Period</th><th>Days</th><th>Hist VaR</th><th>Param VaR</th>
                </tr></thead><tbody>
            """
            for label in labels:
                c = crisis_results[label]
                crisis_table += f"""
                <tr>
                    <td style="font-weight:600;">{label}</td>
                    <td>{c['days']}</td>
                    <td>{c['hist']:.4f}%</td>
                    <td>{c['param']:.4f}%</td>
                </tr>"""
            crisis_table += "</tbody></table>"
            st.markdown(crisis_table, unsafe_allow_html=True)

            # Insight text
            max_crisis = max(crisis_results, key=lambda k: crisis_results[k]['hist'])
            st.info(
                f"📌 **{max_crisis}** had the highest Historical VaR at "
                f"**{crisis_results[max_crisis]['hist']:.4f}%**, "
                f"showing significantly elevated risk during that period."
            )
    else:
        st.warning("No crisis period data available for the selected date range.")

    # ══════════════════════════════════════════════════════════
    # SECTION 4 — MONTE CARLO SIMULATION
    # ══════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">🎲 Monte Carlo Simulation</div>',
                unsafe_allow_html=True)

    with st.spinner(f"Running {n_sim:,} Monte Carlo simulations..."):
        mc_vars, mc_weights_all, min_idx = run_monte_carlo(
            df_returns, available_cols, confidence, n_sim
        )

    min_var_val = mc_vars[min_idx]
    min_weights = mc_weights_all[min_idx]
    eq_var = list(models.values())[0]['var_pct']  # Historical VaR for equal-weight

    col_mc_hist, col_mc_pie = st.columns([3, 2])

    with col_mc_hist:
        fig_mc = go.Figure()
        fig_mc.add_trace(go.Histogram(
            x=mc_vars,
            nbinsx=80,
            marker_color='rgba(52, 211, 153, 0.55)',
            marker_line=dict(width=0.5, color='rgba(52,211,153,0.8)'),
            name='Simulated VaR',
        ))
        fig_mc.add_vline(
            x=min_var_val, line_width=2.5, line_dash="dash",
            line_color="#fb7185",
            annotation_text=f"Min VaR: {min_var_val:.4f}%",
            annotation_font=dict(size=12, color='#fb7185'),
        )
        fig_mc.add_vline(
            x=mc_vars.mean(), line_width=2.5, line_dash="dash",
            line_color="#fbbf24",
            annotation_text=f"Mean: {mc_vars.mean():.4f}%",
            annotation_font=dict(size=12, color='#fbbf24'),
        )
        fig_mc.add_vline(
            x=eq_var, line_width=2.5, line_dash="dash",
            line_color="#6c63ff",
            annotation_text=f"Equal-Wt: {eq_var:.4f}%",
            annotation_font=dict(size=12, color='#6c63ff'),
        )
        fig_mc.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=420,
            margin=dict(l=40, r=20, t=60, b=40),
            title=dict(
                text=f"Monte Carlo VaR Distribution ({n_sim:,} Portfolios)",
                font=dict(size=16, color='#e0e0ff'),
            ),
            xaxis=dict(title="VaR (%)", gridcolor='rgba(255,255,255,0.06)'),
            yaxis=dict(title="Frequency", gridcolor='rgba(255,255,255,0.06)'),
            showlegend=False,
        )
        st.plotly_chart(fig_mc, use_container_width=True)

    with col_mc_pie:
        st.markdown("##### 🏆 Minimum VaR Portfolio")

        # Stats
        mc_stats_html = f"""
        <div class="kpi-container" style="flex-direction: column; gap: 0.6rem;">
            <div class="kpi-card" style="padding: 0.8rem 1.2rem;">
                <div class="kpi-label">Min VaR</div>
                <div class="kpi-value" style="font-size:1.2rem;">{min_var_val:.4f}%</div>
            </div>
            <div class="kpi-card" style="padding: 0.8rem 1.2rem;">
                <div class="kpi-label">Mean VaR (all sims)</div>
                <div class="kpi-value blue" style="font-size:1.2rem;">{mc_vars.mean():.4f}%</div>
            </div>
            <div class="kpi-card" style="padding: 0.8rem 1.2rem;">
                <div class="kpi-label">Max VaR</div>
                <div class="kpi-value rose" style="font-size:1.2rem;">{mc_vars.max():.4f}%</div>
            </div>
        </div>
        """
        st.markdown(mc_stats_html, unsafe_allow_html=True)

        # Pie chart
        clean_names = [s.replace('_Return', '').replace('_return', '').strip()
                       for s in available_cols]
        fig_pie = go.Figure(go.Pie(
            labels=clean_names,
            values=min_weights,
            hole=0.45,
            marker_colors=COLOR_PALETTE[:N],
            textinfo='label+percent',
            textfont=dict(size=11, color='white'),
            hovertemplate='%{label}: %{percent}<extra></extra>',
        ))
        fig_pie.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor='rgba(0,0,0,0)',
            height=320,
            margin=dict(l=10, r=10, t=30, b=10),
            title=dict(
                text="Optimal Weights",
                font=dict(size=14, color='#e0e0ff'),
            ),
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    # SECTION 5 — CORRELATION HEATMAP
    # ══════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">🔗 Stock Return Correlations</div>',
                unsafe_allow_html=True)

    corr_matrix = df_returns.corr()
    clean_labels = [STOCK_NAMES[i] if i < len(STOCK_NAMES)
                    else available_cols[i].replace('_Return', '').replace('_return', '').strip()
                    for i in range(len(available_cols))]

    fig_heat = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=clean_labels,
        y=clean_labels,
        colorscale=[
            [0, '#0f0c29'],
            [0.25, '#302b63'],
            [0.5, '#6c63ff'],
            [0.75, '#a78bfa'],
            [1, '#34d399'],
        ],
        zmin=-0.2, zmax=1,
        text=np.round(corr_matrix.values, 3),
        texttemplate='%{text:.2f}',
        textfont=dict(size=11, color='white'),
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.4f}<extra></extra>',
        colorbar=dict(
            title="ρ",
            titlefont=dict(color='#e0e0ff'),
            tickfont=dict(color='#a0a0a0'),
        ),
    ))
    fig_heat.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=520,
        margin=dict(l=100, r=20, t=50, b=100),
        title=dict(
            text="Pairwise Return Correlation Matrix",
            font=dict(size=16, color='#e0e0ff'),
        ),
        xaxis=dict(tickangle=45, tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11), autorange='reversed'),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    # SECTION 6 — INDIVIDUAL STOCK ANALYSIS
    # ══════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">📋 Individual Stock VaR</div>',
                unsafe_allow_html=True)

    z_val = norm.ppf(1 - confidence)
    stock_data = []
    for i, col in enumerate(available_cols):
        ret = df_returns[col]
        name = STOCK_NAMES[i] if i < len(STOCK_NAMES) else col
        h_var = abs(np.percentile(ret, (1 - confidence) * 100)) * 100
        p_var = abs(ret.mean() + z_val * ret.std()) * 100
        stock_data.append({
            'Stock': name,
            'Mean Return (%)': f"{ret.mean()*100:.4f}",
            'Volatility (%)': f"{ret.std()*100:.4f}",
            'Historical VaR (%)': f"{h_var:.4f}",
            'Parametric VaR (%)': f"{p_var:.4f}",
        })

    stock_table = """
    <table class="styled-table">
        <thead><tr>
            <th>Stock</th><th>Mean Return</th><th>Volatility</th>
            <th>Historical VaR</th><th>Parametric VaR</th>
        </tr></thead><tbody>
    """
    for row in stock_data:
        stock_table += f"""
        <tr>
            <td style="font-weight:600;">{row['Stock']}</td>
            <td>{row['Mean Return (%)']}</td>
            <td>{row['Volatility (%)']}</td>
            <td>{row['Historical VaR (%)']}</td>
            <td>{row['Parametric VaR (%)']}</td>
        </tr>"""
    stock_table += "</tbody></table>"
    st.markdown(stock_table, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # SECTION 7 — RAW DATA EXPLORER
    # ══════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">🗄️ Data Explorer</div>',
                unsafe_allow_html=True)

    with st.expander("📂 View Raw Return Data", expanded=False):
        st.dataframe(
            df_returns.style.format("{:.6f}"),
            use_container_width=True,
            height=400,
        )

    # ── FOOTER ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:rgba(255,255,255,0.3);font-size:0.8rem;'>"
        "Portfolio VaR Analysis Dashboard · Financial Management Project · "
        "Built with Streamlit & Plotly</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
