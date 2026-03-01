"""
================================================================================
 BLOCK TRADES PAGE
 ─────────────────
 Dedicated page for tracking block trade activity across the portfolio.
 Fetches its own live prices for traded tickers (including exited positions).
================================================================================
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytz
import streamlit as st
import yfinance as yf

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

BLOCK_TRADES_FILE = Path(__file__).resolve().parent.parent / "block_trades.json"
HOLDINGS_FILE = Path(__file__).resolve().parent.parent / "holdings.json"
FALLBACK_FX_RATE = 1.36

logging.getLogger("yfinance").setLevel(logging.CRITICAL)


# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS (shared dark theme)
# ──────────────────────────────────────────────────────────────────────────────

PAGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-base: #080c14;
    --bg-surface: #0f1420;
    --bg-card: #111827;
    --bg-elevated: #1a2235;
    --border-subtle: #1c2536;
    --border-default: #243044;
    --border-hover: #344563;
    --text-primary: #edf2f7;
    --text-secondary: #8896ab;
    --text-muted: #576678;
    --accent-blue: #4f8ff7;
    --accent-green: #2dd4a8;
    --accent-red: #f06060;
    --accent-amber: #f5a623;
    --accent-cyan: #22d3ee;
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-pill: 20px;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* ── Section headers ── */
.section-header {
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--border-subtle);
}

/* ── Title bar ── */
.title-bar {
    margin-bottom: 0.25rem;
}
.title-bar h1 {
    font-size: 2rem;
    font-weight: 800;
    margin: 0 0 0.5rem 0;
    color: var(--text-primary);
    letter-spacing: -0.01em;
}
.title-bar .title-meta {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
}
.title-bar .meta-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-pill);
    padding: 0.3rem 0.7rem;
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--text-secondary);
}
.title-bar .meta-chip .meta-label {
    color: var(--text-muted);
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Metric cards ── */
.metric-row {
    display: flex;
    gap: 0.75rem;
    margin: 0.75rem 0 1.25rem 0;
}
.metric-card {
    flex: 1;
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 1.2rem 1.4rem;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    border-color: var(--border-hover);
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.metric-card .label {
    font-size: 0.65rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}
.metric-card .value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.2;
}
.metric-card .delta {
    font-size: 0.78rem;
    font-weight: 500;
    margin-top: 0.35rem;
    color: var(--text-muted);
}
.metric-card .delta.positive { color: var(--accent-green); }
.metric-card .delta.negative { color: var(--accent-red); }

/* ── Block Trade Cards ── */
.trade-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 1rem 1.25rem;
    margin-bottom: 0.5rem;
    transition: border-color 0.2s ease, transform 0.15s ease;
    border-left: 3px solid transparent;
}
.trade-card:hover {
    border-color: var(--border-hover);
    transform: translateY(-1px);
}
.trade-card.buy  { border-left-color: var(--accent-green); }
.trade-card.sell { border-left-color: var(--accent-red); }

.trade-card .trade-top {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.4rem;
    flex-wrap: wrap;
}
.trade-card .trade-type-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: var(--radius-pill);
    font-weight: 700;
    font-size: 0.72rem;
    letter-spacing: 0.03em;
}
.trade-card .trade-type-badge.buy {
    background: rgba(45, 212, 168, 0.08);
    color: var(--accent-green);
    border: 1px solid rgba(45, 212, 168, 0.25);
}
.trade-card .trade-type-badge.sell {
    background: rgba(240, 96, 96, 0.08);
    color: var(--accent-red);
    border: 1px solid rgba(240, 96, 96, 0.25);
}
.trade-card .trade-symbol {
    font-weight: 700;
    font-size: 0.92rem;
    color: var(--text-primary);
}
.trade-card .trade-name {
    font-size: 0.8rem;
    color: var(--text-secondary);
    font-weight: 400;
}
.trade-card .trade-value {
    font-weight: 600;
    font-size: 0.92rem;
    color: var(--text-primary);
    margin-left: auto;
}
.trade-card .trade-details {
    font-size: 0.8rem;
    color: var(--text-secondary);
    line-height: 1.6;
}
.trade-card .trade-details .detail-label {
    color: var(--text-muted);
}
.trade-card .trade-notes {
    font-size: 0.78rem;
    color: var(--text-secondary);
    font-style: italic;
    margin-top: 0.35rem;
    padding-top: 0.35rem;
    border-top: 1px solid var(--border-subtle);
}
.trade-card .switch-indicator {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 2px 8px;
    border-radius: var(--radius-pill);
    background: rgba(79, 143, 247, 0.08);
    color: var(--accent-blue);
    border: 1px solid rgba(79, 143, 247, 0.25);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.trade-card .pnl-badge {
    font-weight: 600;
    font-size: 0.8rem;
}
.trade-date-header {
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 1.25rem 0 0.5rem 0;
    padding-left: 0.25rem;
}

/* ── Sidebar styling ── */
section[data-testid="stSidebar"] {
    background: #060a12;
    border-right: 1px solid var(--border-subtle);
}
section[data-testid="stSidebar"] .stButton > button {
    background: var(--bg-card);
    color: var(--text-secondary);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-sm);
    font-weight: 500;
    transition: all 0.2s ease;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    border-color: var(--accent-blue);
    color: var(--accent-blue);
    background: rgba(79, 143, 247, 0.05);
}

hr {
    border: none;
    border-top: 1px solid var(--border-subtle);
    margin: 2rem 0;
}

.footer-text {
    text-align: center;
    font-size: 0.7rem;
    color: var(--text-muted);
    padding: 1.5rem 0;
    letter-spacing: 0.02em;
}

/* ── Data table ── */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    overflow: hidden;
}
div[data-testid="stDataFrame"] table {
    font-size: 0.85rem;
}

/* ── Custom scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-default); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--border-hover); }
</style>
"""


# ──────────────────────────────────────────────────────────────────────────────
# DATA LAYER
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_block_trades() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Load block trades JSON. Returns (meta, trades_df, switches_df)."""
    if not BLOCK_TRADES_FILE.exists():
        return {}, pd.DataFrame(), pd.DataFrame()
    with open(BLOCK_TRADES_FILE, "r") as f:
        data = json.load(f)
    meta = data.get("_meta", {})
    trades_df = pd.DataFrame(data.get("trades", []))
    switches_df = pd.DataFrame(data.get("switches", []))
    if not trades_df.empty:
        trades_df["date"] = pd.to_datetime(trades_df["date"], errors="coerce")
        trades_df["market_value"] = pd.to_numeric(
            trades_df["market_value"], errors="coerce"
        )
        trades_df["shares"] = pd.to_numeric(trades_df["shares"], errors="coerce")
        trades_df["price"] = pd.to_numeric(trades_df["price"], errors="coerce")
    return meta, trades_df, switches_df


@st.cache_data(ttl=120)
def get_fx_rate() -> float:
    try:
        hist = yf.Ticker("USDCAD=X").history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return FALLBACK_FX_RATE


@st.cache_data(ttl=120)
def fetch_trade_prices(tickers: list[str]) -> pd.DataFrame:
    """Fetch live prices for block trade tickers (including exited positions)."""
    if not tickers:
        return pd.DataFrame()
    # Filter out fund codes (FID670, FID2210, etc.) that aren't on exchanges
    tradeable = [t for t in tickers if not t.startswith("FID")]
    if not tradeable:
        return pd.DataFrame()
    try:
        raw = yf.download(tradeable, period="5d", auto_adjust=True,
                          progress=False, threads=True)
    except Exception:
        return pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tradeable[0]})
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(-1)
    return prices.dropna(axis=1, how="all").ffill()


@st.cache_data(ttl=300)
def load_holdings_for_reference() -> pd.DataFrame:
    """Load holdings for cross-referencing portfolio weights."""
    if not HOLDINGS_FILE.exists():
        return pd.DataFrame()
    with open(HOLDINGS_FILE, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data["holdings"])
    df["weight"] = df["market_value"] / df["market_value"].sum()
    return df


# ──────────────────────────────────────────────────────────────────────────────
# RENDERING
# ──────────────────────────────────────────────────────────────────────────────

def render_metrics(trades_df: pd.DataFrame, switches_df: pd.DataFrame,
                   fx_rate: float) -> str:
    """Build HTML for the block trade summary metric cards."""
    def to_cad(row):
        mv = row.get("market_value")
        if pd.isna(mv) or mv is None:
            return 0
        return mv * fx_rate if row.get("currency") == "USD" else mv

    buys = trades_df[trades_df["type"] == "buy"]
    sells = trades_df[trades_df["type"] == "sell"]

    total_bought = buys.apply(to_cad, axis=1).sum() if not buys.empty else 0
    total_sold = sells.apply(to_cad, axis=1).sum() if not sells.empty else 0
    net_flow = total_bought - total_sold
    net_class = "positive" if net_flow >= 0 else "negative"
    net_arrow = "▲" if net_flow >= 0 else "▼"

    active_switches = 0
    if not switches_df.empty and "status" in switches_df.columns:
        active_switches = len(switches_df[switches_df["status"] != "completed"])

    num_trades = len(trades_df.dropna(subset=["market_value"]))

    return f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="label">Total Bought</div>
            <div class="value" style="color:#2dd4a8;">
                ${total_bought/1e6:.2f}M
            </div>
        </div>
        <div class="metric-card">
            <div class="label">Total Sold</div>
            <div class="value" style="color:#f06060;">
                ${total_sold/1e6:.2f}M
            </div>
        </div>
        <div class="metric-card">
            <div class="label">Net Flow</div>
            <div class="value">
                ${abs(net_flow)/1e6:.2f}M
            </div>
            <div class="delta {net_class}">{net_arrow} {"Net Buying" if net_flow >= 0 else "Net Selling"}</div>
        </div>
        <div class="metric-card">
            <div class="label">Trades / Switches</div>
            <div class="value">
                {num_trades} <span style="color:#576678; font-size:0.9rem;">/</span>
                <span style="color:#4f8ff7;">{len(switches_df) if not switches_df.empty else 0}</span>
            </div>
        </div>
    </div>
    """


def render_trade_card(trade: dict, switch_label: str | None = None,
                      current_price: float | None = None,
                      portfolio_weight: float | None = None) -> str:
    """Build HTML for a single block trade card."""
    t = trade["type"]
    type_label = "BUY" if t == "buy" else "SELL"

    # Market value display
    mv = trade.get("market_value")
    if mv and not pd.isna(mv):
        if mv >= 1_000_000:
            mv_str = f"${mv/1e6:.2f}M"
        else:
            mv_str = f"${mv/1e3:.0f}K"
    else:
        mv_str = ""

    # P&L since trade
    pnl_html = ""
    price = trade.get("price")
    shares = trade.get("shares")
    if (current_price and price and shares
            and not pd.isna(price) and not pd.isna(shares)):
        pnl_per_share = current_price - price
        pnl_total = pnl_per_share * shares
        pnl_pct = (pnl_per_share / price) * 100
        if t == "sell":
            pnl_total = -pnl_total
            pnl_pct = -pnl_pct
        pnl_color = "#2dd4a8" if pnl_total >= 0 else "#f06060"
        pnl_arrow = "▲" if pnl_total >= 0 else "▼"
        pnl_html = (
            f'<span class="pnl-badge" style="color:{pnl_color};">'
            f'&nbsp;&middot;&nbsp; {pnl_arrow} ${abs(pnl_total):,.0f} ({pnl_pct:+.1f}%) since trade'
            f'</span>'
        )

    # Switch badge
    switch_html = ""
    if switch_label:
        switch_html = f'<span class="switch-indicator">&#x1f504; {switch_label}</span>'

    # Portfolio weight badge (if this ticker is still held)
    weight_html = ""
    if portfolio_weight is not None:
        weight_html = (
            f'<span style="background:var(--bg-card); border:1px solid var(--border-subtle); '
            f'border-radius:20px; padding:2px 8px; font-size:0.7rem; '
            f'color:var(--text-secondary); font-weight:500;">'
            f'Portfolio: {portfolio_weight:.1f}%</span>'
        )

    # Details line
    details_parts = []
    if shares and not pd.isna(shares):
        details_parts.append(
            f'<span class="detail-label">Shares:</span> {int(shares):,}'
        )
    if price and not pd.isna(price):
        currency = trade.get("currency", "")
        details_parts.append(
            f'<span class="detail-label">Price:</span> ${price:,.2f} {currency}'
        )
    if current_price:
        details_parts.append(
            f'<span class="detail-label">Current:</span> ${current_price:,.2f}'
        )
    details_line = " &nbsp;&middot;&nbsp; ".join(details_parts)
    if pnl_html:
        details_line += f" {pnl_html}"

    # Notes
    notes_html = ""
    notes = trade.get("notes")
    if notes and str(notes).strip():
        notes_html = f'<div class="trade-notes">{notes}</div>'

    # Name (truncated)
    name = trade.get("name", "")
    if name and len(name) > 50:
        name = name[:47] + "..."

    return f"""
    <div class="trade-card {t}">
        <div class="trade-top">
            <span class="trade-type-badge {t}">{type_label}</span>
            <span class="trade-symbol">{trade.get('symbol', '')}</span>
            <span class="trade-name">{name}</span>
            {switch_html}
            {weight_html}
            <span class="trade-value">{mv_str}</span>
        </div>
        <div class="trade-details">{details_line}</div>
        {notes_html}
    </div>
    """


def make_flow_chart(trades_df: pd.DataFrame, fx_rate: float):
    """Horizontal bar chart showing buy/sell flow by ticker."""
    def to_cad(row):
        mv = row.get("market_value")
        if pd.isna(mv) or mv is None:
            return 0
        return mv * fx_rate if row.get("currency") == "USD" else mv

    # Only trades with market value
    valid = trades_df.dropna(subset=["market_value"]).copy()
    if valid.empty:
        return None

    valid["value_cad"] = valid.apply(to_cad, axis=1)
    # Make sells negative
    valid.loc[valid["type"] == "sell", "value_cad"] *= -1

    grouped = valid.groupby("symbol")["value_cad"].sum().sort_values()

    colors = ["#2dd4a8" if v >= 0 else "#f06060" for v in grouped.values]

    fig = go.Figure(go.Bar(
        x=grouped.values,
        y=grouped.index,
        orientation="h",
        marker=dict(color=colors, line=dict(color="#080c14", width=1)),
        hovertemplate="<b>%{y}</b><br>$%{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text="NET TRADE FLOW BY TICKER",
            font=dict(size=11, color="#576678", family="Inter", weight=700),
            x=0.5, xanchor="center",
        ),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=max(200, len(grouped) * 50 + 80),
        margin=dict(l=100, r=20, t=50, b=30),
        xaxis=dict(
            gridcolor="#1c2536", zerolinecolor="#243044",
            tickfont=dict(color="#576678", size=11),
            tickformat="$,.0f",
        ),
        yaxis=dict(
            tickfont=dict(color="#edf2f7", size=12, family="Inter"),
        ),
        showlegend=False,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# PAGE
# ──────────────────────────────────────────────────────────────────────────────

try:
    st.set_page_config(
        page_title="Block Trades",
        page_icon="🔄",
        layout="wide",
    )
except st.errors.StreamlitAPIException:
    pass  # Already set by main app.py in multipage mode

st.markdown(PAGE_CSS, unsafe_allow_html=True)

# ── Load data ──
bt_meta, trades_df, switches_df = load_block_trades()
fx_rate = get_fx_rate()
holdings_df = load_holdings_for_reference()

if trades_df.empty:
    st.warning("No block trades data found. Place `block_trades.json` in the project root.")
    st.stop()

# Fetch live prices for ALL block trade tickers (including exited positions)
trade_tickers = trades_df["symbol"].dropna().unique().tolist()
with st.spinner("Fetching live prices for traded tickers..."):
    prices = fetch_trade_prices(trade_tickers)

# Build holdings weight lookup
weight_map = {}
if not holdings_df.empty:
    weight_map = dict(zip(holdings_df["ticker"], holdings_df["weight"] * 100))

# ── Sidebar ──
with st.sidebar:
    st.markdown("### Block Trades")
    st.caption(f"USD/CAD: {fx_rate:.4f}")
    st.markdown("---")
    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.caption(
        f"Period: {bt_meta.get('period_start', '—')} to {bt_meta.get('period_end', '—')}  \n"
        f"Last updated: {bt_meta.get('last_updated', '—')}  \n"
        f"Source: {bt_meta.get('source', '—')}"
    )

# ── Period label ──
period_label = ""
ps = bt_meta.get("period_start")
if ps:
    try:
        start = datetime.strptime(ps, "%Y-%m-%d")
        period_label = f" — {start.strftime('%B %Y')}"
    except ValueError:
        pass

now_et = datetime.now(pytz.timezone("US/Eastern"))
refresh_time = now_et.strftime("%I:%M %p ET")

# ══════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════

num_real_trades = len(trades_df.dropna(subset=["market_value"]))

st.markdown(
    f"""
    <div class="title-bar">
        <h1>Block Trades{period_label}</h1>
        <div class="title-meta">
            <span class="meta-chip">
                <span class="meta-label">Trades</span>
                {num_real_trades}
            </span>
            <span class="meta-chip">
                <span class="meta-label">Switches</span>
                <span style="color:#38bdf8;">{len(switches_df) if not switches_df.empty else 0}</span>
            </span>
            <span class="meta-chip">
                <span class="meta-label">FX</span>
                USD/CAD {fx_rate:.4f}
            </span>
            <span class="meta-chip">
                <span class="meta-label">Updated</span>
                {bt_meta.get('last_updated', '—')}
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════
# SUMMARY METRICS
# ══════════════════════════════════════════════════════════════════════════

st.markdown(render_metrics(trades_df, switches_df, fx_rate), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# FLOW CHART + TRADE TABLE (side by side)
# ══════════════════════════════════════════════════════════════════════════

col_chart, col_table = st.columns([2, 3], gap="large")

with col_chart:
    st.markdown('<div class="section-header">Trade Flow</div>', unsafe_allow_html=True)
    flow_fig = make_flow_chart(trades_df, fx_rate)
    if flow_fig:
        st.plotly_chart(flow_fig, use_container_width=True)

with col_table:
    st.markdown('<div class="section-header">Trade Summary</div>', unsafe_allow_html=True)

    # Build a clean summary table
    summary_rows = []
    for _, t in trades_df.iterrows():
        if pd.isna(t.get("market_value")):
            continue
        ticker = t["symbol"]
        current = None
        if ticker in prices.columns:
            series = prices[ticker].dropna()
            if not series.empty:
                current = float(series.iloc[-1])

        pnl_pct = None
        if current and t["price"] and not pd.isna(t["price"]):
            pnl_pct = ((current - t["price"]) / t["price"]) * 100
            if t["type"] == "sell":
                pnl_pct = -pnl_pct

        summary_rows.append({
            "Date": t["date"].strftime("%m/%d") if pd.notna(t["date"]) else "—",
            "Type": t["type"].upper(),
            "Ticker": ticker,
            "Shares": t["shares"],
            "Trade Price": t["price"],
            "Current Price": current,
            "P&L %": pnl_pct,
            "Value (CAD)": t["market_value"] * (fx_rate if t["currency"] == "USD" else 1),
        })

    if summary_rows:
        sum_df = pd.DataFrame(summary_rows)

        def color_pnl(val):
            if pd.isna(val):
                return "color: #576678"
            return "color: #2dd4a8" if val >= 0 else "color: #f06060"

        def color_type(val):
            return "color: #2dd4a8" if val == "BUY" else "color: #f06060"

        styled_sum = (
            sum_df.style
            .format({
                "Shares":        "{:,.0f}",
                "Trade Price":   lambda x: f"${x:,.2f}" if pd.notna(x) else "—",
                "Current Price": lambda x: f"${x:,.2f}" if pd.notna(x) else "—",
                "P&L %":         lambda x: f"{x:+.1f}%" if pd.notna(x) else "—",
                "Value (CAD)":   "${:,.0f}",
            })
            .map(color_pnl, subset=["P&L %"])
            .map(color_type, subset=["Type"])
        )
        st.dataframe(styled_sum, use_container_width=True, height=300, hide_index=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════
# TRADE CARDS (grouped by date)
# ══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">Trade Activity</div>', unsafe_allow_html=True)

# Build switch label lookup
switch_labels = {}
if not switches_df.empty:
    for _, sw in switches_df.iterrows():
        switch_labels[sw["switch_id"]] = sw.get("label", "Switch")

# Filter to displayable trades (have at least a symbol)
displayable = trades_df[trades_df["symbol"].notna()].copy()
displayable = displayable.sort_values("date", ascending=False, na_position="last")

# Group by date and render cards
prev_date = None
for _, trade in displayable.iterrows():
    date_val = trade.get("date")
    if pd.notna(date_val):
        date_str = date_val.strftime("%B %d, %Y")
    else:
        date_str = "Pending"

    if date_str != prev_date:
        st.markdown(
            f'<div class="trade-date-header">{date_str}</div>',
            unsafe_allow_html=True,
        )
        prev_date = date_str

    # Switch label
    sw_id = trade.get("switch_id")
    sw_label = switch_labels.get(sw_id) if sw_id else None

    # Current price lookup
    current_price = None
    ticker = trade.get("symbol", "")
    if ticker in prices.columns:
        series = prices[ticker].dropna()
        if not series.empty:
            current_price = float(series.iloc[-1])

    # Portfolio weight lookup
    pw = weight_map.get(ticker)

    html = render_trade_card(trade.to_dict(), sw_label, current_price, pw)
    st.markdown(html, unsafe_allow_html=True)

# ── Footer ──
st.markdown(
    f'<div class="footer-text">'
    f'Last refreshed: {refresh_time} &middot; '
    f'Live prices via Yahoo Finance &middot; '
    f'Period: {bt_meta.get("period_start", "—")} to {bt_meta.get("period_end", "—")}'
    f'</div>',
    unsafe_allow_html=True,
)
