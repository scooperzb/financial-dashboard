"""
================================================================================
 PORTFOLIO HOLDINGS DASHBOARD
 ─────────────────────────────
 A clean, focused Streamlit dashboard for monitoring top equity holdings.

 HOW TO RUN:
   1. cd C:\\Users\\mathe\\Projects\\financial-dashboard
   2. python -m pip install -r requirements.txt
   3. python -m streamlit run app.py
   — or just double-click run.bat

 LAYOUT:
   1. Holdings Table  → sorted by Value (CAD), live prices & day change
   2. Donut Charts    → Currency Exposure | Sector Allocation
   3. Stock Analysis  → Click any ticker, Claude explains the movement
================================================================================
"""

import json
import logging
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

HOLDINGS_FILE = Path(__file__).parent / "holdings.json"
MODELS_DIR = Path(__file__).parent / "models"
FALLBACK_FX_RATE = 1.36

logging.getLogger("yfinance").setLevel(logging.CRITICAL)


# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM THEME CSS
# ──────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Design Tokens ── */
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

/* ── Hide Streamlit branding ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {
    background: transparent;
}
header[data-testid="stHeader"] .stActionButton {
    visibility: visible !important;
}

/* ── Main container ── */
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

/* ── Hero Section ── */
.hero {
    padding: 0.5rem 0 0.75rem 0;
}
.hero-top {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.15rem;
}
.hero h1 {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0;
}
.hero-status {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--text-muted);
}
.hero-status .dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    display: inline-block;
}
.hero-status .dot.open {
    background: var(--accent-green);
    box-shadow: 0 0 8px rgba(45, 212, 168, 0.35);
}
.hero-status .dot.closed { background: var(--accent-red); }

.hero-value {
    font-size: 3rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    line-height: 1.15;
    margin: 0.15rem 0;
}
.hero-change {
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
}
.hero-change.positive { color: var(--accent-green); }
.hero-change.negative { color: var(--accent-red); }
.hero-change .pnl-label {
    color: var(--text-muted);
    font-weight: 400;
    font-size: 0.85rem;
}

.hero-meta {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-bottom: 0.5rem;
}
.hero-meta .chip {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-pill);
    padding: 0.3rem 0.7rem;
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--text-secondary);
    transition: border-color 0.2s;
}
.hero-meta .chip:hover {
    border-color: var(--border-default);
}
.hero-meta .chip .label {
    color: var(--text-muted);
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Top/Bottom movers strip ── */
.movers-strip {
    display: flex;
    gap: 0.4rem;
    margin: 0.4rem 0 0.2rem 0;
    flex-wrap: wrap;
}
.mover-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.2rem 0.6rem;
    border-radius: var(--radius-pill);
    font-size: 0.72rem;
    font-weight: 600;
    border: 1px solid;
}
.mover-chip.gainer {
    background: rgba(45, 212, 168, 0.06);
    color: var(--accent-green);
    border-color: rgba(45, 212, 168, 0.18);
}
.mover-chip.loser {
    background: rgba(240, 96, 96, 0.06);
    color: var(--accent-red);
    border-color: rgba(240, 96, 96, 0.18);
}
/* ── Mover grid (clickable buttons) ── */
.mover-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 0.5rem;
    margin: 0.75rem 0;
}
.mover-btn {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 0.65rem 0.85rem;
    cursor: pointer;
    transition: all 0.2s ease;
    text-align: left;
}
.mover-btn:hover {
    border-color: var(--border-hover);
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    transform: translateY(-1px);
}
.mover-btn .mover-ticker {
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--text-primary);
}
.mover-btn .mover-pct {
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 0.15rem;
}
.mover-btn .mover-name {
    font-size: 0.65rem;
    color: var(--text-muted);
    margin-top: 0.15rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.mover-btn.gainer { border-left: 3px solid var(--accent-green); }
.mover-btn.gainer .mover-pct { color: var(--accent-green); }
.mover-btn.loser { border-left: 3px solid var(--accent-red); }
.mover-btn.loser .mover-pct { color: var(--accent-red); }

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

/* ── Data table refinements ── */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    overflow: hidden;
}
div[data-testid="stDataFrame"] table {
    font-size: 0.85rem;
}

/* ── Analysis cards ── */
.analysis-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
}
.analysis-card .analysis-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.75rem;
}
.analysis-card .ticker-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: var(--radius-pill);
    font-weight: 700;
    font-size: 0.75rem;
    letter-spacing: 0.03em;
}
.analysis-card .analysis-body {
    font-size: 0.88rem;
    line-height: 1.7;
    color: var(--text-secondary);
}
.analysis-card .analysis-body strong {
    color: var(--text-primary);
    font-weight: 600;
}

/* ── Analysis search select ── */
.analysis-search {
    margin-bottom: 1rem;
}
.analysis-section-label {
    font-size: 0.65rem;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}

/* ── Hero mover buttons ── */
div[data-testid="stHorizontalBlock"] .stButton > button {
    font-size: 0.78rem;
    padding: 0.35rem 0.5rem;
    white-space: nowrap;
}

/* ── 52-Week Range Bar ── */
.range-bar-container {
    margin: 0.5rem 0;
}
.range-bar {
    position: relative;
    height: 6px;
    background: linear-gradient(to right, var(--accent-red), var(--accent-amber), var(--accent-green));
    border-radius: 3px;
    margin: 0.5rem 0 0.25rem 0;
}
.range-bar .marker {
    position: absolute;
    top: -5px;
    width: 16px; height: 16px;
    background: var(--text-primary);
    border: 2px solid var(--bg-card);
    border-radius: 50%;
    transform: translateX(-50%);
    box-shadow: 0 1px 4px rgba(0,0,0,0.4);
}
.range-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.65rem;
    color: var(--text-muted);
    font-weight: 500;
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

/* ── Dividers ── */
hr {
    border: none;
    border-top: 1px solid var(--border-subtle);
    margin: 2rem 0;
}

/* ── Footer ── */
.footer-text {
    text-align: center;
    font-size: 0.7rem;
    color: var(--text-muted);
    padding: 1.5rem 0;
    letter-spacing: 0.02em;
}

/* ── Tabs styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: var(--bg-card);
    border-radius: var(--radius-md);
    padding: 4px;
    border: 1px solid var(--border-subtle);
    margin-bottom: 1rem;
    overflow-x: auto;
    flex-wrap: nowrap;
    -webkit-overflow-scrolling: touch;
}
.stTabs [data-baseweb="tab"] {
    padding: 0.55rem 1.1rem;
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-muted);
    border-radius: var(--radius-sm);
    border: none;
    white-space: nowrap;
    flex-shrink: 0;
    transition: color 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-secondary);
}
.stTabs [aria-selected="true"] {
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
}

/* ── FX Popover sizing ── */
[data-testid="stPopover"] > div {
    min-width: 480px;
    max-height: 85vh;
    overflow-y: auto;
}

/* ── FX Impact card (inside popover) ── */
.fx-impact {
    margin-top: 0.75rem;
    padding: 0.85rem 1rem;
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
}
.fx-impact .fx-label {
    font-size: 0.62rem;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}
.fx-impact .fx-value {
    font-size: 1.15rem;
    font-weight: 700;
    line-height: 1.3;
}
.fx-impact .fx-detail {
    font-size: 0.75rem;
    color: var(--text-secondary);
    margin-top: 0.3rem;
    line-height: 1.6;
}
.fx-impact .fx-explainer {
    font-size: 0.7rem;
    color: var(--text-muted);
    margin-top: 0.4rem;
    padding-top: 0.4rem;
    border-top: 1px solid var(--border-subtle);
    line-height: 1.5;
}

/* ── Custom scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-default); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--border-hover); }

/* ── Multiselect / filter styling ── */
div[data-baseweb="select"] {
    font-size: 0.85rem;
}
</style>
"""


# ──────────────────────────────────────────────────────────────────────────────
# DATA LAYER
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_holdings() -> tuple[dict, pd.DataFrame]:
    if not HOLDINGS_FILE.exists():
        st.error(f"Holdings file not found at `{HOLDINGS_FILE}`.")
        st.stop()
    with open(HOLDINGS_FILE, "r") as f:
        data = json.load(f)
    meta = data.get("_meta", {})
    df = pd.DataFrame(data["holdings"])
    df["weight"] = df["market_value"] / df["market_value"].sum()
    return meta, df


@st.cache_data(ttl=300)
def load_all_models() -> list:
    """Load all model JSON files from the models/ directory.
    Returns a list of (meta, constituents_df) tuples, sorted by tab_name.
    """
    models = []
    if not MODELS_DIR.exists():
        st.warning(f"Models directory not found: {MODELS_DIR}")
        return models
    json_files = sorted(MODELS_DIR.glob("*.json"))
    if not json_files:
        st.warning(f"No JSON files found in {MODELS_DIR}")
        return models
    for fpath in json_files:
        try:
            with open(fpath, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            meta = data.get("_meta", {})
            meta.setdefault("tab_name", meta.get("model_name", fpath.stem))
            df = pd.DataFrame(data.get("constituents", []))
            if not df.empty:
                models.append((meta, df))
        except Exception as e:
            st.error(f"Error loading model {fpath.name}: {e}")
    return models


@st.cache_data(ttl=120)
def get_fx_rate() -> float:
    try:
        hist = yf.Ticker("USDCAD=X").history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return FALLBACK_FX_RATE


@st.cache_data(ttl=300)
def fetch_fx_history(period: str = "30d") -> pd.DataFrame:
    """Fetch historical USD/CAD exchange rates for the FX popover chart."""
    try:
        hist = yf.Ticker("USDCAD=X").history(period=period)
        if not hist.empty:
            return hist[["Close"]].dropna()
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=120)
def fetch_prices(tickers: list[str]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    try:
        raw = yf.download(tickers, period="5d", auto_adjust=True,
                          progress=False, threads=True)
    except Exception as e:
        st.warning(f"Yahoo Finance error: {e}")
        return pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(-1)
    return prices.dropna(axis=1, how="all").ffill()


# ──────────────────────────────────────────────────────────────────────────────
# FUNDAMENTALS
# ──────────────────────────────────────────────────────────────────────────────

# Tickers that are ETFs/commodities/crypto — skip for P/E, EPS, etc.
_NON_EQUITY_TICKERS = {
    "GLD", "CGL.TO", "IBIT", "ETHA", "BGU-U.TO",
    "VIU.TO", "IDIV-B.TO",
}


@st.cache_data(ttl=600, show_spinner=False)
def fetch_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """
    Fetch key fundamental metrics from yfinance .info for each ticker.
    Returns a DataFrame indexed by ticker with columns:
      trailingPE, forwardPE, dividendYield, beta,
      fiftyTwoWeekHigh, fiftyTwoWeekLow, marketCap, sector_yf
    """
    rows = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception:
            info = {}

        div_yield = info.get("dividendYield")
        if div_yield is None:
            div_yield = info.get("trailingAnnualDividendYield")

        # Normalize: yfinance should return decimals (0.03 = 3%) but some
        # tickers return the value already as a percentage (3.0) or even
        # absurd numbers.  Clamp to a sane range.
        if div_yield is not None:
            try:
                div_yield = float(div_yield)
                # If > 1 it was likely returned as a percentage — convert
                if div_yield > 1.0:
                    div_yield = div_yield / 100.0
                # Cap at 25% — anything higher is a data error
                if div_yield > 0.25 or div_yield < 0:
                    div_yield = None
            except (TypeError, ValueError):
                div_yield = None

        rows.append({
            "ticker": ticker,
            "trailingPE": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
            "dividendYield": div_yield,
            "beta": info.get("beta"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
            "marketCap": info.get("marketCap"),
        })
    return pd.DataFrame(rows).set_index("ticker")


# ── Macro data ──────────────────────────────────────────────────────────────

MACRO_TICKERS = ["^VIX", "^TNX", "^FVX", "CL=F", "GC=F", "DX-Y.NYB", "ZAG.TO"]
FED_FUNDS_RATE = 4.33  # Effective federal funds rate — update after FOMC decisions
BOC_RATE = 2.75        # Bank of Canada overnight rate — update after BoC decisions


@st.cache_data(ttl=120, show_spinner=False)
def fetch_macro_data() -> dict:
    """Fetch macro indicators (VIX, yields, oil, gold, DXY) via yfinance."""
    try:
        df = yf.download(
            MACRO_TICKERS, period="6mo", auto_adjust=True,
            progress=False, threads=True,
        )
        if df.empty:
            return {}

        # yf.download with multiple tickers returns MultiIndex columns (field, ticker)
        closes = df["Close"].ffill()

        current = {}
        changes = {}
        for t in MACRO_TICKERS:
            if t not in closes.columns:
                continue
            series = closes[t].dropna()
            if len(series) < 2:
                continue
            current[t] = float(series.iloc[-1])
            prev = float(series.iloc[-2])
            if prev != 0:
                changes[t] = ((current[t] - prev) / prev) * 100
            else:
                changes[t] = 0.0

        return {"prices": closes, "current": current, "changes": changes}
    except Exception:
        return {}


def compute_portfolio_metrics(table: pd.DataFrame, fundamentals: pd.DataFrame,
                              holdings: pd.DataFrame) -> dict:
    """
    Compute weighted-average portfolio metrics.
    Only includes equities (skips ETFs/commodities) for P/E and EPS-based metrics.
    """
    # Merge fundamentals with table by ticker
    merged = table.merge(fundamentals, left_on="Ticker", right_index=True, how="left")
    total_value = merged["Value (CAD)"].sum()
    if total_value == 0:
        return {}

    merged["weight"] = merged["Value (CAD)"] / total_value

    # Filter equities only (exclude ETFs, commodities, crypto)
    equities = merged[~merged["Ticker"].isin(_NON_EQUITY_TICKERS)]

    # --- Weighted P/E (trailing) ---
    pe_valid = equities.dropna(subset=["trailingPE"])
    pe_valid = pe_valid[pe_valid["trailingPE"] > 0]
    if not pe_valid.empty:
        pe_weight_sum = pe_valid["weight"].sum()
        weighted_pe = (pe_valid["trailingPE"] * pe_valid["weight"]).sum() / pe_weight_sum
    else:
        weighted_pe = None

    # --- Weighted Forward P/E ---
    fpe_valid = equities.dropna(subset=["forwardPE"])
    fpe_valid = fpe_valid[fpe_valid["forwardPE"] > 0]
    if not fpe_valid.empty:
        fpe_weight_sum = fpe_valid["weight"].sum()
        weighted_fpe = (fpe_valid["forwardPE"] * fpe_valid["weight"]).sum() / fpe_weight_sum
    else:
        weighted_fpe = None

    # --- Weighted Dividend Yield ---
    # Treat holdings with no yield data as 0% yield (not excluded).
    # This gives the true portfolio-level blended yield.
    # Extra safety: clip 0–0.25 (0–25%) to catch any surviving outliers.
    merged["div_clean"] = merged["dividendYield"].fillna(0).clip(lower=0, upper=0.25)
    weighted_yield = (merged["div_clean"] * merged["weight"]).sum()
    if weighted_yield <= 0:
        weighted_yield = None

    # --- Weighted Beta ---
    # Weight against full portfolio — holdings without beta assumed beta=1 (market)
    merged["beta_clean"] = merged["beta"].fillna(1.0)
    weighted_beta = (merged["beta_clean"] * merged["weight"]).sum()
    if weighted_beta <= 0:
        weighted_beta = None

    # --- 52-Week Positioning ---
    # What % of holdings are near 52w high vs 52w low
    range_data = merged.dropna(subset=["fiftyTwoWeekHigh", "fiftyTwoWeekLow", "Price"])
    near_high = 0  # within 5% of 52w high
    near_low = 0   # within 5% of 52w low
    positions_with_range = 0
    avg_52w_position = []  # 0 = at low, 100 = at high

    for _, r in range_data.iterrows():
        hi = r["fiftyTwoWeekHigh"]
        lo = r["fiftyTwoWeekLow"]
        px = r["Price"]
        if hi > lo and hi > 0:
            positions_with_range += 1
            pct_in_range = (px - lo) / (hi - lo) * 100
            avg_52w_position.append((pct_in_range, r["weight"]))
            if px >= hi * 0.95:
                near_high += 1
            if px <= lo * 1.05:
                near_low += 1

    if avg_52w_position:
        weighted_52w = sum(p * w for p, w in avg_52w_position) / sum(w for _, w in avg_52w_position)
    else:
        weighted_52w = None

    # --- Coverage stats ---
    pe_coverage = len(pe_valid) / len(equities) * 100 if len(equities) > 0 else 0

    return {
        "weighted_pe": weighted_pe,
        "weighted_fpe": weighted_fpe,
        "weighted_yield": weighted_yield,
        "weighted_beta": weighted_beta,
        "weighted_52w_position": weighted_52w,
        "near_52w_high": near_high,
        "near_52w_low": near_low,
        "pe_coverage": pe_coverage,
        "total_value": total_value,
    }


# ──────────────────────────────────────────────────────────────────────────────
# STOCK MOVEMENT ANALYSIS (Claude API)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def _get_sp500_change() -> tuple[str, float]:
    """Fetch today's S&P 500 change via ^GSPC."""
    try:
        spy = yf.Ticker("^GSPC")
        hist = spy.history(period="5d")
        if len(hist) >= 2:
            last = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2])
            chg = ((last / prev) - 1) * 100
            return ("up" if chg >= 0 else "down", abs(chg))
    except Exception:
        pass
    return ("flat", 0.0)


@st.cache_data(ttl=3600, show_spinner=False)
def query_stock_movement(ticker: str, company_name: str,
                         day_change: float, price: float,
                         sector: str = "—") -> str:
    """Ask Claude (with web search) for the latest reason behind a stock move."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "ANTHROPIC_API_KEY not configured — add it to your environment variables."
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        direction = "up" if day_change >= 0 else "down"
        sp500_direction, sp500_change = _get_sp500_change()
        prompt = (
            f"You are a financial news analyst. Your job is to identify the most "
            f"likely catalyst for a stock's move today.\n\n"
            f"**Stock context:**\n"
            f"- Company: {company_name} ({ticker})\n"
            f"- Sector: {sector}\n"
            f"- Today's move: {direction} {abs(day_change):.2f}% (${price:.2f})\n"
            f"- Market context: S&P 500 is {sp500_direction} {sp500_change:.2f}% today\n\n"
            f"**Instructions:**\n"
            f"1. Search for news about {ticker} from today and the last 24 hours. "
            f"Prioritize earnings releases, guidance changes, analyst actions, "
            f"regulatory developments, management changes, and M&A activity.\n"
            f"2. If no company-specific catalyst is found, search for sector or macro "
            f"developments that would explain the move (e.g., rate decisions, sector "
            f"rotation, commodity price shifts).\n"
            f"3. Attribute the move to the most specific catalyst you can identify. "
            f"Do not default to \"broad market weakness/strength\" unless the stock's "
            f"move is closely tracking the index and no idiosyncratic driver exists.\n\n"
            f"**Response format (output ONLY this, no preamble or thinking):**\n"
            f"- **Catalyst:** [One-line summary of the primary driver]\n"
            f"- **Detail:** [2-4 sentences with specifics — dates, figures, analyst "
            f"names, or policy details where available]\n"
            f"- **Confidence:** [High / Medium / Low] based on how directly the "
            f"catalyst explains the magnitude and direction of the move\n\n"
            f"IMPORTANT: Start your response directly with \"- **Catalyst:**\". "
            f"Do not include any introductory text, thinking, or commentary before the structured response."
        )
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 5,
            }],
            messages=[{"role": "user", "content": prompt}],
        )
        # Extract text blocks from the response (web search returns mixed content)
        parts = []
        for block in message.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return " ".join(parts).strip() if parts else "No analysis available."
    except Exception as e:
        return f"Error querying Claude: {e}"


# ──────────────────────────────────────────────────────────────────────────────
# COMPUTATIONS
# ──────────────────────────────────────────────────────────────────────────────

def build_table(holdings: pd.DataFrame, prices: pd.DataFrame,
                fx_rate: float) -> pd.DataFrame:
    rows = []
    for _, h in holdings.iterrows():
        ticker = h["ticker"]
        is_usd = h["currency"] == "USD"
        row = {"Ticker": ticker, "Shares": h["quantity"],
               "Sector": h.get("sector", "—"), "Currency": h["currency"]}
        if ticker in prices.columns and len(prices[ticker].dropna()) >= 2:
            series = prices[ticker].dropna()
            last = float(series.iloc[-1])
            prev = float(series.iloc[-2])
            row["Price"] = last
            row["Day Change %"] = ((last / prev) - 1) * 100
            value_local = h["quantity"] * last
            row["Value (CAD)"] = value_local * fx_rate if is_usd else value_local
        else:
            row["Price"] = None
            row["Day Change %"] = None
            row["Value (CAD)"] = h["market_value"]
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values("Value (CAD)", ascending=False).reset_index(drop=True)


def build_model_comparison(model_df: pd.DataFrame,
                           table: pd.DataFrame,
                           holdings: pd.DataFrame) -> pd.DataFrame:
    """Compare model target weights against actual portfolio weights."""
    total_value = table["Value (CAD)"].sum()
    name_map = holdings.set_index("ticker")["name"].to_dict()

    # Compute actual weight per ticker
    actual_weights = {}
    if total_value > 0:
        for _, row in table.iterrows():
            actual_weights[row["Ticker"]] = (row["Value (CAD)"] / total_value) * 100

    rows = []

    # Model constituents
    for _, m in model_df.iterrows():
        ticker = m["ticker"]
        target = m["target_pct"]
        actual = actual_weights.pop(ticker, 0.0)
        divergence = actual - target
        status = "Matched" if actual > 0 else "Model Only"
        rows.append({
            "Ticker": ticker,
            "Name": m["name"],
            "Target %": target,
            "Actual %": round(actual, 2),
            "Divergence %": round(divergence, 2),
            "Status": status,
        })

    result = pd.DataFrame(rows)
    return result.sort_values("Divergence %", ascending=True).reset_index(drop=True)


def build_model_composition(model_df: pd.DataFrame,
                            holdings: pd.DataFrame) -> tuple:
    """Aggregate model target_pct by sector and currency.

    Uses a three-tier lookup:
      1. sector/currency fields in the model JSON itself
      2. Fallback to holdings.json lookup by ticker
      3. Infer currency from ticker suffix (.TO = CAD, else USD); sector = "Other"

    Returns (sector_series, currency_series) mapping category → sum of target_pct.
    """
    holdings_lookup = holdings.set_index("ticker")[["sector", "currency"]].to_dict("index")

    rows = []
    for _, m in model_df.iterrows():
        ticker = m["ticker"]
        target = m["target_pct"]

        # Tier 1: from model JSON
        sector = m.get("sector") if "sector" in m.index else None
        currency = m.get("currency") if "currency" in m.index else None

        # Tier 2: from holdings
        if not sector and ticker in holdings_lookup:
            sector = holdings_lookup[ticker].get("sector")
        if not currency and ticker in holdings_lookup:
            currency = holdings_lookup[ticker].get("currency")

        # Tier 3: infer
        if not currency:
            currency = "CAD" if ticker.endswith(".TO") else "USD"
        if not sector:
            sector = "Other"

        rows.append({"sector": sector, "currency": currency, "target_pct": target})

    comp = pd.DataFrame(rows)
    sector_agg = comp.groupby("sector")["target_pct"].sum().sort_values(ascending=False)
    currency_agg = comp.groupby("currency")["target_pct"].sum().sort_values(ascending=False)

    return sector_agg, currency_agg


# ──────────────────────────────────────────────────────────────────────────────
# CHARTS
# ──────────────────────────────────────────────────────────────────────────────

SECTOR_COLORS = {
    "Financials": "#818cf8", "Technology": "#34d399", "Energy": "#f87171",
    "Utilities": "#c084fc", "Industrials": "#fb923c", "Healthcare": "#22d3ee",
    "Consumer Staples": "#f472b6", "Consumer Discretionary": "#a3e635",
    "Real Estate": "#e879f9", "Telecommunications": "#fbbf24",
    "Commodities": "#fcd34d", "Digital Assets": "#f59e0b", "ETF - Equity": "#94a3b8",
    "Cash": "#64748b",
}

def make_donut(labels, values, title, colors=None, center_text=None,
               hover_format=None):
    total = sum(values)
    # Build custom legend labels: "Label  XX.X%"
    pcts = [(v / total * 100) if total else 0 for v in values]
    legend_labels = [f"{lbl}  {p:.1f}%" for lbl, p in zip(labels, pcts)]

    if hover_format == "pct":
        ht = "<b>%{label}</b><br>%{value:.2f}%<extra></extra>"
    else:
        ht = "<b>%{label}</b><br>$%{value:,.0f}<extra></extra>"

    fig = go.Figure(go.Pie(
        labels=legend_labels, values=values, hole=0.6,
        marker=dict(
            colors=colors,
            line=dict(color="#080c14", width=2),
        ) if colors else dict(line=dict(color="#080c14", width=2)),
        textinfo="none",
        hovertemplate=ht,
        sort=True, direction="clockwise",
    ))

    if center_text is None:
        center_text = f"<b>${total/1e6:.0f}M</b>"

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=11, color="#576678", family="Inter", weight=700),
            x=0.5, xanchor="center", y=0.97,
        ),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=500, margin=dict(l=10, r=10, t=40, b=10),
        showlegend=True,
        legend=dict(
            font=dict(size=11, color="#8896ab", family="Inter"),
            orientation="h",
            yanchor="top", y=-0.02,
            xanchor="center", x=0.5,
            itemwidth=30,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            traceorder="normal",
        ),
        annotations=[dict(
            text=center_text,
            x=0.5, y=0.5, font=dict(size=22, color="#edf2f7", family="Inter"),
            showarrow=False,
        )],
    )
    return fig


def make_fx_chart(fx_df: pd.DataFrame) -> go.Figure:
    """Build a compact 30-day USD/CAD line chart for the FX popover."""
    dates = fx_df.index
    closes = fx_df["Close"]
    y_min = closes.min()
    y_max = closes.max()
    y_pad = (y_max - y_min) * 0.15 or 0.005  # 15% padding, min 0.005

    fig = go.Figure()

    # Area fill from y_min baseline (not zero)
    fig.add_trace(go.Scatter(
        x=dates, y=closes, mode="lines",
        line=dict(color="#4f8ff7", width=2),
        fill="tonexty", fillcolor="rgba(79, 143, 247, 0.08)",
        hovertemplate="<b>%{x|%b %d}</b><br>USD/CAD: %{y:.4f}<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220, margin=dict(l=0, r=0, t=10, b=0),
        font=dict(color="#8896ab", family="Inter", size=11),
        hovermode="x unified",
        showlegend=False,
        xaxis=dict(
            showgrid=False, zeroline=False,
            tickformat="%b %d",
            linecolor="#243044",
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#1c2536", zeroline=False,
            tickformat=".4f",
            side="right",
            range=[y_min - y_pad, y_max + y_pad],
        ),
    )
    return fig


def make_macro_chart(dates, values, title: str, color: str = "#4f8ff7",
                     y_format: str = ".2f", height: int = 260) -> go.Figure:
    """Reusable 6-month line+area chart for macro indicators."""
    y_min, y_max = values.min(), values.max()
    y_pad = (y_max - y_min) * 0.15 or 0.5

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=values, mode="lines",
        line=dict(color=color, width=2),
        fill="tonexty",
        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)",
        hovertemplate=f"<b>%{{x|%b %d}}</b><br>{title}: %{{y:{y_format}}}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height, margin=dict(l=0, r=0, t=10, b=0),
        font=dict(color="#8896ab", family="Inter", size=11),
        hovermode="x unified",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, tickformat="%b %d",
                   linecolor="#243044"),
        yaxis=dict(showgrid=True, gridcolor="#1c2536", zeroline=False,
                   tickformat=y_format, side="right",
                   range=[y_min - y_pad, y_max + y_pad]),
    )
    return fig


def make_yield_chart(dates, y10, y5) -> go.Figure:
    """Yield curve chart: 10Y vs 5Y with spread shading."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=y10, mode="lines", name="10Y",
        line=dict(color="#4f8ff7", width=2),
        hovertemplate="<b>%{x|%b %d}</b><br>10Y: %{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=y5, mode="lines", name="5Y",
        line=dict(color="#22d3ee", width=2),
        fill="tonexty",
        fillcolor="rgba(79, 143, 247, 0.06)",
        hovertemplate="<b>%{x|%b %d}</b><br>5Y: %{y:.2f}%<extra></extra>",
    ))

    y_all = pd.concat([y10, y5])
    y_min, y_max = y_all.min(), y_all.max()
    y_pad = (y_max - y_min) * 0.15 or 0.2

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=260, margin=dict(l=0, r=0, t=10, b=0),
        font=dict(color="#8896ab", family="Inter", size=11),
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center",
                    font=dict(size=10)),
        xaxis=dict(showgrid=False, zeroline=False, tickformat="%b %d",
                   linecolor="#243044"),
        yaxis=dict(showgrid=True, gridcolor="#1c2536", zeroline=False,
                   tickformat=".2f", side="right",
                   range=[y_min - y_pad, y_max + y_pad]),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# STREAMLIT APP
# ──────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Portfolio Dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Load everything ──
    meta, holdings = load_holdings()
    fx_rate = get_fx_rate()
    all_tickers = holdings["ticker"].tolist()

    with st.spinner("Fetching live prices..."):
        prices = fetch_prices(all_tickers)

    if prices.empty:
        st.error("Could not fetch market data. Check your internet connection.")
        st.stop()

    # Sidebar
    fetched = set(prices.columns)
    failed = set(all_tickers) - fetched
    if failed:
        st.sidebar.warning(
            f"No data for {len(failed)} ticker(s): "
            + ", ".join(sorted(failed)[:8])
            + ("..." if len(failed) > 8 else "")
        )

    with st.sidebar:
        st.markdown("### Portfolio Dashboard")
        st.caption(f"USD/CAD: {fx_rate:.4f}")
        st.markdown("---")
        if st.button("Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.markdown("---")
        st.caption(
            f"Report: {meta.get('report_date', '—')}  \n"
            f"Positions: {meta.get('total_positions', len(holdings))}  \n"
            f"Source: Yahoo Finance"
        )

    # ── Build the master table ──
    table = build_table(holdings, prices, fx_rate)

    # Count gainers / losers
    valid = table.dropna(subset=["Day Change %"])
    gainers = len(valid[valid["Day Change %"] > 0])
    losers = len(valid[valid["Day Change %"] < 0])

    # Portfolio value & day PnL
    total_value = table["Value (CAD)"].sum()
    if not valid.empty and total_value > 0:
        weighted_day_chg = (
            (valid["Value (CAD)"] * valid["Day Change %"]).sum() / total_value
        )
        day_pnl = total_value * weighted_day_chg / 100
    else:
        weighted_day_chg = 0
        day_pnl = 0

    # ══════════════════════════════════════════════════════════════════════
    # HEADER
    # ══════════════════════════════════════════════════════════════════════

    # Determine market status (NYSE hours: 9:30–16:00 ET, Mon–Fri)
    from datetime import datetime
    import pytz
    now_et = datetime.now(pytz.timezone("US/Eastern"))
    is_weekday = now_et.weekday() < 5
    market_open_time = now_et.replace(hour=9, minute=30, second=0)
    market_close_time = now_et.replace(hour=16, minute=0, second=0)
    market_is_open = is_weekday and market_open_time <= now_et <= market_close_time
    market_dot = "open" if market_is_open else "closed"
    market_label = "Market Open" if market_is_open else "Market Closed"

    # Top 3 gainers & losers for mover chips (also used for clickable buttons later)
    sorted_up = valid.nlargest(3, "Day Change %")
    sorted_down = valid.nsmallest(3, "Day Change %")

    gainer_chips = "".join(
        f'<span class="mover-chip gainer">'
        f'{r["Ticker"]} ▲{r["Day Change %"]:.2f}%</span>'
        for _, r in sorted_up.iterrows() if r["Day Change %"] > 0
    )
    loser_chips = "".join(
        f'<span class="mover-chip loser">'
        f'{r["Ticker"]} ▼{abs(r["Day Change %"]):.2f}%</span>'
        for _, r in sorted_down.iterrows() if r["Day Change %"] < 0
    )

    # Initialize session state for analysis ticker
    if "analysis_ticker" not in st.session_state:
        st.session_state["analysis_ticker"] = None

    report_date = meta.get("report_date", "—")
    num_positions = meta.get("total_positions", len(holdings))
    refresh_time = now_et.strftime("%I:%M %p ET")

    pnl_sign = "positive" if day_pnl >= 0 else "negative"
    pnl_arrow = "▲" if day_pnl >= 0 else "▼"

    # ── FX impact on portfolio ──
    # USD/CAD ▲ (USD strengthens) → USD holdings worth more in CAD → tailwind
    # USD/CAD ▼ (USD weakens)     → USD holdings worth less in CAD → headwind
    usd_value = table[table["Currency"] == "USD"]["Value (CAD)"].sum()
    usd_pct = (usd_value / total_value * 100) if total_value > 0 else 0
    fx_hist_full = fetch_fx_history("1y")
    fx_30d_chg = 0.0
    fx_impact_30d = 0.0
    if not fx_hist_full.empty:
        cutoff_30d = fx_hist_full.index[-1] - pd.Timedelta(days=30)
        fx_30d_slice = fx_hist_full[fx_hist_full.index >= cutoff_30d]
        if len(fx_30d_slice) >= 2:
            fx_first = float(fx_30d_slice["Close"].iloc[0])
            fx_last = float(fx_30d_slice["Close"].iloc[-1])
            fx_30d_chg = ((fx_last - fx_first) / fx_first) * 100
            fx_impact_30d = usd_value * fx_30d_chg / 100

    if abs(fx_30d_chg) < 0.01:
        fx_wind_label = "Neutral"
        fx_wind_color = "#576678"
    elif fx_30d_chg > 0:
        fx_wind_label = "Tailwind"
        fx_wind_color = "#2dd4a8"
    else:
        fx_wind_label = "Headwind"
        fx_wind_color = "#f06060"

    fx_chip_arrow = "▲" if fx_30d_chg > 0 else "▼" if fx_30d_chg < 0 else "—"

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-top">
                <h1>Portfolio Holdings</h1>
                <span class="hero-status">
                    <span class="dot {market_dot}"></span> {market_label}
                </span>
            </div>
            <div class="hero-value">${total_value:,.0f}</div>
            <div class="hero-change {pnl_sign}">
                {pnl_arrow} ${abs(day_pnl):,.0f} ({weighted_day_chg:+.2f}%)
                <span class="pnl-label">&nbsp;today</span>
            </div>
            <div class="hero-meta">
                <span class="chip">
                    <span class="label">As of</span> {report_date}
                </span>
                <span class="chip">
                    <span class="label">Positions</span> {num_positions}
                </span>
                <span class="chip">
                    <span style="color:#2dd4a8;">{gainers} ▲</span>
                    &nbsp;/&nbsp;
                    <span style="color:#f06060;">{losers} ▼</span>
                </span>
                <span class="chip">
                    <span class="label">FX</span> USD/CAD {fx_rate:.4f}
                    &nbsp;<span style="color:{fx_wind_color}; font-weight:700;">{fx_chip_arrow} {fx_wind_label}</span>
                </span>
            </div>
            <div class="movers-strip">
                {gainer_chips}{loser_chips}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── FX Popover with chart + portfolio impact ──
    with st.popover(f"💱 USD/CAD {fx_rate:.4f}", use_container_width=False):
        if not fx_hist_full.empty:
            # Period selector
            period_opt = st.radio(
                "Period", ["30D", "90D", "6M", "1Y"],
                horizontal=True, key="fx_period", label_visibility="collapsed",
            )
            period_days = {"30D": 30, "90D": 90, "6M": 180, "1Y": 365}
            cutoff = fx_hist_full.index[-1] - pd.Timedelta(
                days=period_days[period_opt]
            )
            fx_slice = fx_hist_full[fx_hist_full.index >= cutoff]

            if len(fx_slice) >= 2:
                p_closes = fx_slice["Close"]
                p_hi = p_closes.max()
                p_lo = p_closes.min()
                p_first = float(p_closes.iloc[0])
                p_last = float(p_closes.iloc[-1])
                p_chg = ((p_last - p_first) / p_first) * 100
                p_impact = usd_value * p_chg / 100
                p_color = "#2dd4a8" if p_chg >= 0 else "#f06060"
                p_arrow = "▲" if p_chg >= 0 else "▼"
                p_direction = "strengthened" if p_chg > 0 else "weakened"
                p_wind = "Tailwind" if p_chg > 0 else "Headwind" if p_chg < 0 else "Neutral"

                # Portfolio impact card (shown first so it's always visible)
                st.markdown(
                    f"""
                    <div class="fx-impact">
                        <div class="fx-label">Portfolio FX Impact ({period_opt})</div>
                        <div class="fx-value" style="color:{p_color};">
                            {p_arrow} ${abs(p_impact)/1e6:.2f}M &nbsp;
                            <span style="font-size:0.85rem;">{p_wind}</span>
                        </div>
                        <div class="fx-detail">
                            USD {p_direction} {abs(p_chg):.2f}% vs CAD over {period_opt.lower()}.<br>
                            Your USD exposure: <b style="color:#edf2f7;">${usd_value/1e6:.1f}M</b>
                            ({usd_pct:.0f}% of portfolio).
                        </div>
                        <div class="fx-explainer">
                            USD/CAD ▲ = USD worth
                            <b style="color:#2dd4a8;">more</b> in CAD
                            <span style="color:#2dd4a8;">→ tailwind</span>
                            &nbsp;&nbsp;
                            USD/CAD ▼ = USD worth
                            <b style="color:#f06060;">less</b> in CAD
                            <span style="color:#f06060;">→ headwind</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Chart
                st.plotly_chart(
                    make_fx_chart(fx_slice), use_container_width=True,
                    key="fx_popover_chart",
                )

                # Stats row
                st.markdown(
                    f"""
                    <div style="display:flex; justify-content:space-between;
                                font-size:0.78rem; color:#8896ab; padding:0 0.2rem;">
                        <span>High: <b style="color:#edf2f7;">{p_hi:.4f}</b></span>
                        <span>Low: <b style="color:#edf2f7;">{p_lo:.4f}</b></span>
                        <span>Change:
                            <b style="color:{p_color};">
                                {p_arrow} {abs(p_chg):.2f}%
                            </b>
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.caption("Not enough data for selected period")
        else:
            st.caption("FX history unavailable")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 1 — HOLDINGS / MODEL TABS
    # ══════════════════════════════════════════════════════════════════════

    all_models = load_all_models()

    tab_names = ["Holdings", "Macro"] + [m[0].get("tab_name", "Model") for m in all_models]
    tabs = st.tabs(tab_names)

    # ── Holdings tab ──
    with tabs[0]:
        st.markdown('<div class="section-header">Holdings</div>',
                    unsafe_allow_html=True)

        display = table[["Ticker", "Shares", "Price", "Day Change %",
                          "Value (CAD)", "Sector"]].copy()

        all_tickers_sorted = sorted(display["Ticker"].unique())
        all_sectors_sorted = sorted(display["Sector"].dropna().unique())
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            sel_tickers = st.multiselect(
                "Filter by ticker", all_tickers_sorted,
                placeholder="Type to search tickers...",
            )
        with filter_col2:
            sel_sectors = st.multiselect(
                "Filter by sector", all_sectors_sorted,
                placeholder="Type to search sectors...",
            )
        if sel_tickers:
            display = display[display["Ticker"].isin(sel_tickers)]
        if sel_sectors:
            display = display[display["Sector"].isin(sel_sectors)]

        def color_day_change(val):
            if pd.isna(val):
                return "color: #576678"
            return "color: #2dd4a8" if val >= 0 else "color: #f06060"

        styled = (
            display.style
            .format({
                "Shares":       "{:,.0f}",
                "Price":        lambda x: f"${x:,.2f}" if pd.notna(x) else "—",
                "Day Change %": lambda x: f"{x:+.2f}%" if pd.notna(x) else "—",
                "Value (CAD)":  "${:,.0f}",
            })
            .map(color_day_change, subset=["Day Change %"])
        )

        st.dataframe(styled, use_container_width=True, height=600, hide_index=True)

    # ── Macro tab ──
    with tabs[1]:
        st.markdown(
            '<div class="section-header">Macro Overview</div>',
            unsafe_allow_html=True,
        )
        with st.spinner("Fetching macro data..."):
            macro = fetch_macro_data()

        if not macro:
            st.caption("Macro data unavailable — try refreshing.")
        else:
            mc = macro.get("current", {})
            md = macro.get("changes", {})
            prices_df = macro.get("prices", pd.DataFrame())

            # Helper for delta color
            def _delta_html(val, fmt=".2f", suffix="%", invert=False):
                if val is None:
                    return '<span style="color:var(--text-muted);">—</span>'
                c = "#2dd4a8" if (val < 0 if invert else val > 0) else "#f06060" if (val > 0 if invert else val < 0) else "var(--text-muted)"
                arrow = "▲" if val > 0 else "▼" if val < 0 else ""
                return f'<span style="color:{c};">{arrow} {abs(val):{fmt}}{suffix}</span>'

            # Yield spread
            y10 = mc.get("^TNX")
            y5 = mc.get("^FVX")
            spread = (y10 - y5) if y10 is not None and y5 is not None else None
            spread_color = "#2dd4a8" if spread and spread > 0 else "#f06060" if spread and spread < 0 else "var(--text-muted)"

            # VIX color
            vix_val = mc.get("^VIX")
            if vix_val is not None:
                vix_color = "#2dd4a8" if vix_val < 20 else "#f5a623" if vix_val < 30 else "#f06060"
            else:
                vix_color = "var(--text-primary)"

            # ── Rate & Index Cards ──
            st.markdown(
                f"""
                <div class="metric-row">
                    <div class="metric-card">
                        <div class="label">Fed Funds Rate</div>
                        <div class="value">{FED_FUNDS_RATE:.2f}%</div>
                        <div class="delta" style="color:var(--text-muted);">FOMC target</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">US 10Y Yield</div>
                        <div class="value">{f'{y10:.2f}%' if y10 is not None else '—'}</div>
                        <div class="delta">{_delta_html(md.get("^TNX"), ".2f", "%")}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">US 5Y Yield</div>
                        <div class="value">{f'{y5:.2f}%' if y5 is not None else '—'}</div>
                        <div class="delta">{_delta_html(md.get("^FVX"), ".2f", "%")}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Yield Spread (10Y-5Y)</div>
                        <div class="value" style="color:{spread_color};">{f'{spread:+.2f}%' if spread is not None else '—'}</div>
                        <div class="delta" style="color:var(--text-muted);">{'Normal' if spread and spread > 0 else 'Inverted' if spread and spread < 0 else '—'}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">BoC Overnight Rate</div>
                        <div class="value">{BOC_RATE:.2f}%</div>
                        <div class="delta" style="color:var(--text-muted);">Bank of Canada</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Market & Commodity Cards ──
            oil = mc.get("CL=F")
            gold = mc.get("GC=F")
            zag = mc.get("ZAG.TO")
            st.markdown(
                f"""
                <div class="metric-row">
                    <div class="metric-card">
                        <div class="label">VIX</div>
                        <div class="value" style="color:{vix_color};">{f'{vix_val:.1f}' if vix_val is not None else '—'}</div>
                        <div class="delta">{_delta_html(md.get("^VIX"), ".1f", "")}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">DXY (US Dollar)</div>
                        <div class="value">{f'{mc.get("DX-Y.NYB", 0):.2f}' if "DX-Y.NYB" in mc else '—'}</div>
                        <div class="delta">{_delta_html(md.get("DX-Y.NYB"), ".2f", "%")}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">WTI Crude Oil</div>
                        <div class="value">{f'${oil:.2f}' if oil is not None else '—'}</div>
                        <div class="delta">{_delta_html(md.get("CL=F"), ".2f", "%")}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Gold</div>
                        <div class="value">{f'${gold:,.2f}' if gold is not None else '—'}</div>
                        <div class="delta">{_delta_html(md.get("GC=F"), ".2f", "%")}</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">CAD Bonds (ZAG.TO)</div>
                        <div class="value">{f'${zag:.2f}' if zag is not None else '—'}</div>
                        <div class="delta">{_delta_html(md.get("ZAG.TO"), ".2f", "%")}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("---")

            # ── Charts ──
            chart_row1_l, chart_row1_r = st.columns(2)

            with chart_row1_l:
                st.markdown(
                    '<div class="section-header">Yield Curve · 6 Month</div>',
                    unsafe_allow_html=True,
                )
                if (not prices_df.empty
                        and "^TNX" in prices_df.columns
                        and "^FVX" in prices_df.columns):
                    s10 = prices_df["^TNX"].dropna()
                    s5 = prices_df["^FVX"].dropna()
                    common = s10.index.intersection(s5.index)
                    if len(common) > 5:
                        fig_yc = make_yield_chart(common, s10.loc[common], s5.loc[common])
                        st.plotly_chart(fig_yc, use_container_width=True, config={"displayModeBar": False})
                    else:
                        st.caption("Insufficient yield data.")
                else:
                    st.caption("Yield data unavailable.")

            with chart_row1_r:
                st.markdown(
                    '<div class="section-header">VIX · 6 Month</div>',
                    unsafe_allow_html=True,
                )
                if not prices_df.empty and "^VIX" in prices_df.columns:
                    vix_s = prices_df["^VIX"].dropna()
                    if len(vix_s) > 5:
                        fig_vix = make_macro_chart(
                            vix_s.index, vix_s, "VIX",
                            color="#f5a623", y_format=".1f",
                        )
                        st.plotly_chart(fig_vix, use_container_width=True, config={"displayModeBar": False})
                    else:
                        st.caption("Insufficient VIX data.")
                else:
                    st.caption("VIX data unavailable.")

            chart_row2_l, chart_row2_r = st.columns(2)

            with chart_row2_l:
                st.markdown(
                    '<div class="section-header">WTI Crude · 6 Month</div>',
                    unsafe_allow_html=True,
                )
                if not prices_df.empty and "CL=F" in prices_df.columns:
                    oil_s = prices_df["CL=F"].dropna()
                    if len(oil_s) > 5:
                        fig_oil = make_macro_chart(
                            oil_s.index, oil_s, "WTI Crude",
                            color="#4f8ff7", y_format="$.2f",
                        )
                        st.plotly_chart(fig_oil, use_container_width=True, config={"displayModeBar": False})
                    else:
                        st.caption("Insufficient oil data.")
                else:
                    st.caption("Oil data unavailable.")

            with chart_row2_r:
                st.markdown(
                    '<div class="section-header">Gold · 6 Month</div>',
                    unsafe_allow_html=True,
                )
                if not prices_df.empty and "GC=F" in prices_df.columns:
                    gold_s = prices_df["GC=F"].dropna()
                    if len(gold_s) > 5:
                        fig_gold = make_macro_chart(
                            gold_s.index, gold_s, "Gold",
                            color="#f5a623", y_format="$,.0f",
                        )
                        st.plotly_chart(fig_gold, use_container_width=True, config={"displayModeBar": False})
                    else:
                        st.caption("Insufficient gold data.")
                else:
                    st.caption("Gold data unavailable.")

            chart_row3_l, chart_row3_r = st.columns(2)

            with chart_row3_l:
                st.markdown(
                    '<div class="section-header">DXY (US Dollar Index) · 6 Month</div>',
                    unsafe_allow_html=True,
                )
                if not prices_df.empty and "DX-Y.NYB" in prices_df.columns:
                    dxy_s = prices_df["DX-Y.NYB"].dropna()
                    if len(dxy_s) > 5:
                        fig_dxy = make_macro_chart(
                            dxy_s.index, dxy_s, "DXY",
                            color="#818cf8", y_format=".2f",
                        )
                        st.plotly_chart(fig_dxy, use_container_width=True, config={"displayModeBar": False})
                    else:
                        st.caption("Insufficient DXY data.")
                else:
                    st.caption("DXY data unavailable.")

            with chart_row3_r:
                st.markdown(
                    '<div class="section-header">CAD Bonds (ZAG.TO) · 6 Month</div>',
                    unsafe_allow_html=True,
                )
                if not prices_df.empty and "ZAG.TO" in prices_df.columns:
                    zag_s = prices_df["ZAG.TO"].dropna()
                    if len(zag_s) > 5:
                        fig_zag = make_macro_chart(
                            zag_s.index, zag_s, "ZAG.TO",
                            color="#2dd4a8", y_format="$.2f",
                        )
                        st.plotly_chart(fig_zag, use_container_width=True, config={"displayModeBar": False})
                    else:
                        st.caption("Insufficient ZAG data.")
                else:
                    st.caption("ZAG data unavailable.")

    # ── Model tabs (dynamic) ──
    for i, (model_meta, model_df) in enumerate(all_models):
        with tabs[i + 2]:
            model_key = f"model_{i}"
            st.caption(
                f"{model_meta.get('model_name', 'Model')} · "
                f"Cash: {model_meta.get('cash_pct', 0)}% · "
                f"{model_meta.get('num_constituents', 0)} constituents"
            )

            comparison = build_model_comparison(model_df, table, holdings)

            # Filters
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                m_sel_tickers = st.multiselect(
                    "Filter by ticker",
                    sorted(comparison["Ticker"].unique()),
                    placeholder="Type to search tickers...",
                    key=f"{model_key}_ticker_filter",
                )
            with m_col2:
                m_sel_status = st.multiselect(
                    "Filter by status",
                    sorted(comparison["Status"].unique()),
                    placeholder="Filter by status...",
                    key=f"{model_key}_status_filter",
                )

            display_model = comparison.copy()
            if m_sel_tickers:
                display_model = display_model[display_model["Ticker"].isin(m_sel_tickers)]
            if m_sel_status:
                display_model = display_model[display_model["Status"].isin(m_sel_status)]

            def color_divergence(val):
                if pd.isna(val) or val == 0:
                    return "color: #576678"
                return "color: #2dd4a8" if val > 0 else "color: #f06060"

            def color_status(val):
                if val == "Model Only":
                    return "color: #f06060"
                if val == "Not in Model":
                    return "color: #f5a623"
                return "color: #8896ab"

            styled_model = (
                display_model.style
                .format({
                    "Target %":     "{:.2f}%",
                    "Actual %":     "{:.2f}%",
                    "Divergence %": lambda x: f"{x:+.2f}%",
                })
                .map(color_divergence, subset=["Divergence %"])
                .map(color_status, subset=["Status"])
            )

            st.dataframe(styled_model, use_container_width=True, height=600,
                          hide_index=True)

            # Summary metrics — dynamic per model
            in_both = len(comparison[comparison["Status"] == "Matched"])
            model_only = len(comparison[comparison["Status"] == "Model Only"])
            total_positions = len(comparison)
            avg_div = comparison["Divergence %"].abs().mean()

            st.markdown(
                f"""
                <div class="metric-row">
                    <div class="metric-card">
                        <div class="label">Total Positions</div>
                        <div class="value">{total_positions}</div>
                        <div class="delta">Model constituents</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Matched Positions</div>
                        <div class="value" style="color:#2dd4a8;">{in_both}</div>
                        <div class="delta">In both model & portfolio</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Model Only</div>
                        <div class="value" style="color:#f06060;">{model_only}</div>
                        <div class="delta">In model but not held</div>
                    </div>
                    <div class="metric-card">
                        <div class="label">Avg. Abs. Divergence</div>
                        <div class="value">{avg_div:.2f}%</div>
                        <div class="delta">Mean |actual - target|</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Model Composition Donuts ──
            sector_agg, currency_agg = build_model_composition(
                model_df, holdings)

            st.markdown(
                '<div class="section-header" style="margin-top:1.5rem;">'
                'Model Target Composition</div>',
                unsafe_allow_html=True,
            )

            m_chart_left, m_chart_right = st.columns([2, 3], gap="large")

            with m_chart_left:
                curr_colors = [
                    "#2dd4a8" if c == "CAD" else "#f06060"
                    for c in currency_agg.index
                ]
                total_equity_pct = model_meta.get("total_equity_pct",
                                                   currency_agg.sum())
                fig_curr = make_donut(
                    currency_agg.index.tolist(),
                    currency_agg.values.tolist(),
                    "CURRENCY EXPOSURE (TARGET)",
                    colors=curr_colors,
                    center_text=f"<b>{total_equity_pct:.1f}%</b>",
                    hover_format="pct",
                )
                st.plotly_chart(fig_curr, use_container_width=True,
                                key=f"{model_key}_curr_donut")

            with m_chart_right:
                # Add cash slice so sector donut sums to 100%
                cash_pct = model_meta.get("cash_pct", 0)
                if cash_pct > 0:
                    sector_with_cash = pd.concat([
                        sector_agg,
                        pd.Series({"Cash": cash_pct}),
                    ])
                else:
                    sector_with_cash = sector_agg

                sector_colors = [
                    SECTOR_COLORS.get(s, "#6b7280")
                    for s in sector_with_cash.index
                ]
                fig_sect = make_donut(
                    sector_with_cash.index.tolist(),
                    sector_with_cash.values.tolist(),
                    "SECTOR ALLOCATION (TARGET)",
                    colors=sector_colors,
                    center_text=f"<b>{total_equity_pct:.1f}%</b>",
                    hover_format="pct",
                )
                st.plotly_chart(fig_sect, use_container_width=True,
                                key=f"{model_key}_sect_donut")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 2 — DONUT CHARTS
    # ══════════════════════════════════════════════════════════════════════

    st.markdown('<div class="section-header">Portfolio Composition</div>',
                unsafe_allow_html=True)

    merged = table.copy()
    merged["Currency"] = holdings["currency"].values
    merged["Sector"] = holdings["sector"].values

    chart_left, chart_right = st.columns([2, 3], gap="large")

    with chart_left:
        curr_agg = merged.groupby("Currency")["Value (CAD)"].sum()
        fig_curr = make_donut(
            curr_agg.index.tolist(), curr_agg.values.tolist(),
            "CURRENCY EXPOSURE", ["#2dd4a8", "#f06060"],
        )
        st.plotly_chart(fig_curr, use_container_width=True)

    with chart_right:
        sect_agg = merged.groupby("Sector")["Value (CAD)"].sum().sort_values(ascending=False)
        sector_colors = [SECTOR_COLORS.get(s, "#6b7280") for s in sect_agg.index]
        fig_sect = make_donut(
            sect_agg.index.tolist(), sect_agg.values.tolist(),
            "SECTOR ALLOCATION", sector_colors,
        )
        st.plotly_chart(fig_sect, use_container_width=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 2.5 — PORTFOLIO METRICS
    # ══════════════════════════════════════════════════════════════════════

    st.markdown('<div class="section-header">Portfolio Metrics</div>',
                unsafe_allow_html=True)
    st.caption("Value-weighted fundamentals · Excludes ETFs and commodities for P/E")

    with st.spinner("Fetching fundamentals..."):
        fundamentals = fetch_fundamentals(all_tickers)

    metrics = compute_portfolio_metrics(table, fundamentals, holdings)

    if metrics:
        w_pe = metrics.get("weighted_pe")
        w_fpe = metrics.get("weighted_fpe")
        w_yield = metrics.get("weighted_yield")
        w_beta = metrics.get("weighted_beta")
        w_52w = metrics.get("weighted_52w_position")
        near_hi = metrics.get("near_52w_high", 0)
        near_lo = metrics.get("near_52w_low", 0)
        pe_cov = metrics.get("pe_coverage", 0)

        # Row 1: Main metrics
        st.markdown(
            f"""
            <div class="metric-row">
                <div class="metric-card">
                    <div class="label">Weighted P/E (Trailing)</div>
                    <div class="value">
                        {f'{w_pe:.1f}x' if w_pe else '—'}
                    </div>
                    <div class="delta">
                        {f'Forward: {w_fpe:.1f}x' if w_fpe else 'Forward: —'}
                        &nbsp;&middot;&nbsp; Coverage: {pe_cov:.0f}%
                    </div>
                </div>
                <div class="metric-card">
                    <div class="label">Weighted Dividend Yield</div>
                    <div class="value" style="color:#2dd4a8;">
                        {f'{w_yield*100:.2f}%' if w_yield else '—'}
                    </div>
                    <div class="delta">
                        {f'~${metrics["total_value"]*w_yield/1e6:.1f}M annual income' if w_yield else ''}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="label">Weighted Beta</div>
                    <div class="value">
                        {f'{w_beta:.2f}' if w_beta else '—'}
                    </div>
                    <div class="delta" style="color:{'#2dd4a8' if w_beta and w_beta < 1 else '#f06060' if w_beta and w_beta > 1 else 'var(--text-muted)'};">
                        {'Lower volatility than market' if w_beta and w_beta < 1 else 'Higher volatility than market' if w_beta and w_beta > 1 else ''}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="label">52-Week Positioning</div>
                    <div class="value">
                        {f'{w_52w:.0f}%' if w_52w is not None else '—'}
                    </div>
                    <div class="delta">
                        {near_hi} near highs &nbsp;&middot;&nbsp; {near_lo} near lows
                    </div>
                    <div class="range-bar-container">
                        <div class="range-bar">
                            <div class="marker" style="left:{w_52w if w_52w is not None else 50}%;"></div>
                        </div>
                        <div class="range-labels">
                            <span>52W Low</span>
                            <span>52W High</span>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Could not compute portfolio metrics.")

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 3 — STOCK MOVEMENT ANALYSIS  (wrapped in @st.fragment so
    # button clicks only rerun this section, keeping scroll position)
    # ══════════════════════════════════════════════════════════════════════

    # Pre-compute data the fragment needs (avoids referencing outer locals
    # that change on full reruns)
    name_map = holdings.set_index("ticker")["name"].to_dict()
    movers = (
        table.dropna(subset=["Day Change %"])
        .assign(abs_chg=lambda df: df["Day Change %"].abs())
        .sort_values("abs_chg", ascending=False)
    )
    movers_idx = movers.set_index("Ticker")
    top_gainers = valid.nlargest(5, "Day Change %")
    top_gainers = top_gainers[top_gainers["Day Change %"] > 0]
    top_losers = valid.nsmallest(5, "Day Change %")
    top_losers = top_losers[top_losers["Day Change %"] < 0]

    @st.fragment
    def _analysis_fragment():
        st.markdown(
            '<div id="analysis-section" class="section-header">'
            'Stock Movement Analysis</div>',
            unsafe_allow_html=True,
        )
        st.caption("Select a ticker to ask Claude why it moved · Cached 1 hour")

        # ── Top movers grid (5 gainers + 5 losers) ──
        if len(top_gainers) > 0:
            st.markdown(
                '<div class="analysis-section-label">🟢 Top Gainers — click to analyze</div>',
                unsafe_allow_html=True,
            )
            g_cols = st.columns(len(top_gainers))
            for idx, (_, r) in enumerate(top_gainers.iterrows()):
                with g_cols[idx]:
                    if st.button(
                        f"{r['Ticker']}\n:green[▲ {r['Day Change %']:.2f}%]",
                        key=f"gain_{r['Ticker']}",
                        use_container_width=True,
                    ):
                        st.session_state["analysis_ticker"] = r["Ticker"]

        if len(top_losers) > 0:
            st.markdown(
                '<div class="analysis-section-label">🔴 Top Losers — click to analyze</div>',
                unsafe_allow_html=True,
            )
            l_cols = st.columns(len(top_losers))
            for idx, (_, r) in enumerate(top_losers.iterrows()):
                with l_cols[idx]:
                    if st.button(
                        f"{r['Ticker']}\n:red[▼ {abs(r['Day Change %']):.2f}%]",
                        key=f"loss_{r['Ticker']}",
                        use_container_width=True,
                    ):
                        st.session_state["analysis_ticker"] = r["Ticker"]

        # ── Search any ticker ──
        all_ticker_options = [""] + movers["Ticker"].tolist()
        ticker_display = {"": "Search all tickers..."}
        for t in movers["Ticker"].tolist():
            chg = movers_idx.loc[t, "Day Change %"]
            arrow = "▲" if chg >= 0 else "▼"
            short_name = name_map.get(t, t)[:30]
            ticker_display[t] = f"{t}  {arrow} {abs(chg):.2f}%  —  {short_name}"

        search_col, _ = st.columns([1, 2])
        with search_col:
            searched = st.selectbox(
                "Or search any holding",
                options=all_ticker_options,
                format_func=lambda x: ticker_display.get(x, x),
                key="ticker_search",
                label_visibility="collapsed",
            )
            if searched:
                st.session_state["analysis_ticker"] = searched

        # ── Display analysis for selected ticker ──
        selected_ticker = st.session_state.get("analysis_ticker")

        if selected_ticker and selected_ticker in movers_idx.index:
            row = movers_idx.loc[selected_ticker]
            day_chg = row["Day Change %"]
            price = row.get("Price", 0)
            sector = row.get("Sector", "—")
            company = name_map.get(selected_ticker, selected_ticker)
            chg_color = "#2dd4a8" if day_chg >= 0 else "#f06060"
            chg_arrow = "▲" if day_chg >= 0 else "▼"

            with st.spinner(f"Asking Claude about {selected_ticker}..."):
                analysis = query_stock_movement(
                    selected_ticker, company, day_chg, price, sector
                )

            st.markdown(
                f"""
                <div class="analysis-card">
                    <div class="analysis-header">
                        <span class="ticker-badge"
                              style="background:{chg_color}18; color:{chg_color}; border:1px solid {chg_color}44;">
                            {selected_ticker} {chg_arrow} {abs(day_chg):.2f}%
                        </span>
                        <span style="color:var(--text-muted); font-size:0.8rem;">
                            {company} · ${price:,.2f}
                        </span>
                    </div>
                    <div class="analysis-body">
                        {analysis}
                    </div>
                </div>
                <script>
                    // Scroll the analysis card into view after render
                    (function() {{
                        var el = document.querySelector('.analysis-card');
                        if (el) el.scrollIntoView({{behavior: 'smooth', block: 'center'}});
                    }})();
                </script>
                """,
                unsafe_allow_html=True,
            )

    _analysis_fragment()

    # ── Footer ──
    st.markdown(
        f'<div class="footer-text">'
        f'Last refreshed: {refresh_time} &middot; '
        f'Report: {meta.get("report_date", "—")} &middot; '
        f'Live prices via Yahoo Finance &middot; '
        f'Analysis via Claude'
        f'</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        st.error(f"Dashboard error: {e}")
        st.code(traceback.format_exc())
