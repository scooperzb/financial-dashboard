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
   3. Market Intel    → Top 10 stories from today's biggest movers
================================================================================
"""

import json
import logging
import re
from pathlib import Path

import nltk
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from textblob import TextBlob

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

# TextBlob requires NLTK corpora — download on first run
for corpus in ["punkt_tab", "punkt"]:
    try:
        nltk.data.find(f"tokenizers/{corpus}")
    except LookupError:
        try:
            nltk.download(corpus, quiet=True)
        except Exception:
            pass

HOLDINGS_FILE = Path(__file__).parent / "holdings.json"
MODEL_FILE = Path(__file__).parent / "model_portfolio.json"
FALLBACK_FX_RATE = 1.36

logging.getLogger("yfinance").setLevel(logging.CRITICAL)


# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM THEME CSS
# ──────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── Hide Streamlit branding ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ── Main container ── */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* ── Section headers ── */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e293b;
}

/* ── Dashboard title bar ── */
.title-bar {
    margin-bottom: 0.25rem;
}
.title-bar h1 {
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.5rem 0;
    color: #f1f5f9;
}
.title-bar .title-meta {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    flex-wrap: wrap;
}
.title-bar .meta-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 0.4rem 0.9rem;
    font-size: 0.9rem;
    font-weight: 500;
    color: #e2e8f0;
}
.title-bar .meta-chip .meta-icon {
    font-size: 1rem;
}
.title-bar .meta-chip .meta-label {
    color: #64748b;
    font-size: 0.75rem;
    font-weight: 400;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-right: 0.3rem;
}
.title-bar .market-status {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.8rem;
    font-weight: 500;
    color: #64748b;
}
.title-bar .market-status .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
}
.title-bar .market-status .dot.open { background: #34d399; box-shadow: 0 0 6px #34d39966; }
.title-bar .market-status .dot.closed { background: #f87171; }

/* ── Top/Bottom movers strip ── */
.movers-strip {
    display: flex;
    gap: 0.6rem;
    margin: 1rem 0 0.5rem 0;
    flex-wrap: wrap;
}
.mover-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.3rem 0.75rem;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 600;
    border: 1px solid;
}
.mover-chip.gainer {
    background: #34d39912;
    color: #34d399;
    border-color: #34d39933;
}
.mover-chip.loser {
    background: #f8717112;
    color: #f87171;
    border-color: #f8717133;
}

/* ── Metric cards ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0 1.5rem 0;
}
.metric-card {
    flex: 1;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
}
.metric-card .label {
    font-size: 0.75rem;
    font-weight: 500;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.35rem;
}
.metric-card .value {
    font-size: 1.65rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1.2;
}
.metric-card .delta {
    font-size: 0.9rem;
    font-weight: 600;
    margin-top: 0.2rem;
}
.metric-card .delta.positive { color: #34d399; }
.metric-card .delta.negative { color: #f87171; }

/* ── Data table refinements ── */
div[data-testid="stDataFrame"] {
    border: 1px solid #1e293b;
    border-radius: 10px;
    overflow: hidden;
}
div[data-testid="stDataFrame"] table {
    font-size: 0.85rem;
}

/* ── News cards ── */
.news-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.2s ease;
}
.news-card:hover {
    border-color: #334155;
}
.news-card .news-top {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.45rem;
    flex-wrap: wrap;
}
.news-card .ticker-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 0.78rem;
    letter-spacing: 0.03em;
}
.news-card .sentiment-badge {
    font-weight: 600;
    font-size: 0.8rem;
}
.news-card .publisher {
    font-size: 0.78rem;
    color: #475569;
    margin-left: auto;
}
.news-card .headline {
    font-size: 0.92rem;
    font-weight: 500;
    line-height: 1.45;
    color: #e2e8f0;
}
.news-card .headline a {
    color: #e2e8f0;
    text-decoration: none;
}
.news-card .headline a:hover {
    color: #38bdf8;
    text-decoration: underline;
}

/* ── 52-Week Range Bar ── */
.range-bar-container {
    margin: 0.5rem 0;
}
.range-bar {
    position: relative;
    height: 8px;
    background: linear-gradient(to right, #f87171, #facc15, #34d399);
    border-radius: 4px;
    margin: 0.5rem 0 0.25rem 0;
}
.range-bar .marker {
    position: absolute;
    top: -4px;
    width: 16px; height: 16px;
    background: #f1f5f9;
    border: 2px solid #0f172a;
    border-radius: 50%;
    transform: translateX(-50%);
}
.range-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.7rem;
    color: #64748b;
}

/* ── Sidebar styling ── */
section[data-testid="stSidebar"] {
    background: #0a0f1a;
    border-right: 1px solid #1e293b;
}
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #1e3a5f 0%, #1e293b 100%);
    color: #e2e8f0;
    border: 1px solid #334155;
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.2s ease;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    border-color: #38bdf8;
    color: #38bdf8;
}

/* ── Dividers ── */
hr {
    border: none;
    border-top: 1px solid #1e293b;
    margin: 1.5rem 0;
}

/* ── Footer ── */
.footer-text {
    text-align: center;
    font-size: 0.75rem;
    color: #475569;
    padding: 1rem 0;
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
def load_model_portfolio() -> tuple[dict, pd.DataFrame]:
    if not MODEL_FILE.exists():
        return {}, pd.DataFrame()
    with open(MODEL_FILE, "r") as f:
        data = json.load(f)
    meta = data.get("_meta", {})
    df = pd.DataFrame(data.get("constituents", []))
    return meta, df


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
# NEWS & RELEVANCE
# ──────────────────────────────────────────────────────────────────────────────

def _build_relevance_keywords(ticker: str, company_name: str) -> list[str]:
    keywords = set()
    base_ticker = ticker.split(".")[0].replace("-", ".").lower()
    keywords.add(base_ticker)
    keywords.add(ticker.lower())
    skip = {"inc", "corp", "ltd", "the", "of", "class", "new", "com", "sub",
            "common", "stock", "clb", "adr", "etf", "trust", "unit",
            "partnership", "reit", "units", "exchangeable", "vtg", "vot"}
    for word in company_name.lower().replace(",", "").replace(".", "").split():
        if len(word) >= 3 and word not in skip:
            keywords.add(word)
    aliases = {
        "alphabet": ["google"], "meta": ["facebook", "instagram"],
        "brookfield": ["brookfield"], "spdr": ["gold", "gld"],
        "ishares": ["bitcoin", "ethereum"],
    }
    for key, extras in aliases.items():
        if key in keywords:
            keywords.update(extras)
    return list(keywords)


def _is_relevant(title: str, summary: str, keywords: list[str]) -> bool:
    text = (title + " " + summary).lower()
    return any(re.search(r'\b' + re.escape(kw) + r'\b', text) for kw in keywords)


@st.cache_data(ttl=3600)
def fetch_news_for_ticker(ticker: str, company_name: str) -> list[dict]:
    try:
        t = yf.Ticker(ticker)
        raw_news = t.news or []
    except Exception:
        return []
    keywords = _build_relevance_keywords(ticker, company_name)
    items = []
    for article in raw_news:
        # Handle both yfinance <1.0 (flat keys) and >=1.0 (nested "content")
        content = article.get("content", article)
        title = content.get("title", article.get("title", ""))
        summary = content.get("summary", article.get("summary", ""))
        if not _is_relevant(title, summary, keywords):
            continue
        click_through = content.get("clickThroughUrl") or {}
        canonical = content.get("canonicalUrl") or {}
        link = (click_through.get("url")
                or canonical.get("url")
                or article.get("link", ""))
        provider = content.get("provider") or {}
        publisher = provider.get("displayName", article.get("publisher", "Unknown"))
        polarity = TextBlob(title).sentiment.polarity
        if polarity > 0.05:
            sentiment = "Bullish"
        elif polarity < -0.05:
            sentiment = "Bearish"
        else:
            sentiment = "Neutral"
        items.append({"title": title, "link": link, "publisher": publisher,
                       "sentiment": sentiment, "polarity": polarity,
                       "ticker": ticker, "company": company_name})
    return items


def get_top_movers_news(table: pd.DataFrame, holdings: pd.DataFrame,
                        top_n_movers: int = 15, top_n_stories: int = 10) -> list[dict]:
    name_map = holdings.set_index("ticker")["name"].to_dict()
    movers = (
        table.dropna(subset=["Day Change %"])
        .assign(**{"abs_chg": lambda df: df["Day Change %"].abs()})
        .sort_values("abs_chg", ascending=False)
        .head(top_n_movers)
    )
    all_news, seen_titles, ticker_counts = [], set(), {}
    for rank, (_, row) in enumerate(movers.iterrows()):
        ticker = row["Ticker"]
        company = name_map.get(ticker, ticker)
        day_chg = row["Day Change %"]
        max_for_ticker = 4 if rank == 0 else 2
        ticker_counts.setdefault(ticker, 0)
        stories = fetch_news_for_ticker(ticker, company)
        for story in stories:
            if ticker_counts[ticker] >= max_for_ticker:
                break
            title_key = story["title"].lower().strip()
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            story["day_change"] = day_chg
            story["abs_change"] = abs(day_chg)
            all_news.append(story)
            ticker_counts[ticker] += 1
    all_news.sort(key=lambda x: x["abs_change"], reverse=True)
    return all_news[:top_n_stories]


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

    # Holdings not in model
    for ticker, actual in actual_weights.items():
        rows.append({
            "Ticker": ticker,
            "Name": name_map.get(ticker, ""),
            "Target %": 0.0,
            "Actual %": round(actual, 2),
            "Divergence %": round(actual, 2),
            "Status": "Not in Model",
        })

    result = pd.DataFrame(rows)
    return result.sort_values("Divergence %", ascending=True).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# CHARTS
# ──────────────────────────────────────────────────────────────────────────────

SECTOR_COLORS = {
    "Financials": "#6366f1", "Technology": "#22c55e", "Energy": "#ef4444",
    "Utilities": "#a855f7", "Industrials": "#f97316", "Healthcare": "#06b6d4",
    "Consumer Staples": "#ec4899", "Consumer Discretionary": "#84cc16",
    "Real Estate": "#d946ef", "Telecommunications": "#eab308",
    "Commodities": "#facc15", "Digital Assets": "#f59e0b", "ETF - Equity": "#6b7280",
}

SENTIMENT_STYLE = {
    "Bullish":  ("▲", "#34d399"),
    "Bearish":  ("▼", "#f87171"),
    "Neutral":  ("—", "#64748b"),
}


def make_donut(labels, values, title, colors=None):
    total = sum(values)
    # Build custom legend labels: "Label  XX.X%"
    pcts = [(v / total * 100) if total else 0 for v in values]
    legend_labels = [f"{lbl}  {p:.1f}%" for lbl, p in zip(labels, pcts)]

    fig = go.Figure(go.Pie(
        labels=legend_labels, values=values, hole=0.55,
        marker=dict(
            colors=colors,
            line=dict(color="#0f172a", width=2),
        ) if colors else dict(line=dict(color="#0f172a", width=2)),
        textinfo="none",
        hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<extra></extra>",
        sort=True, direction="clockwise",
    ))
    fig.update_layout(
        title=dict(
            text=title, font=dict(size=13, color="#94a3b8", family="Inter"),
            x=0.5, xanchor="center", y=0.97,
        ),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=520, margin=dict(l=10, r=10, t=40, b=10),
        showlegend=True,
        legend=dict(
            font=dict(size=12, color="#cbd5e1", family="Inter"),
            orientation="h",
            yanchor="top", y=-0.02,
            xanchor="center", x=0.5,
            itemwidth=30,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            traceorder="normal",
        ),
        annotations=[dict(
            text=f"<b>${total/1e6:.0f}M</b>",
            x=0.5, y=0.5, font=dict(size=20, color="#e2e8f0", family="Inter"),
            showarrow=False,
        )],
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
            f"Source: Yahoo Finance  \n"
            f"News: Hourly refresh"
        )

    # ── Build the master table ──
    table = build_table(holdings, prices, fx_rate)

    # Count gainers / losers
    valid = table.dropna(subset=["Day Change %"])
    gainers = len(valid[valid["Day Change %"] > 0])
    losers = len(valid[valid["Day Change %"] < 0])

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

    # Top 3 gainers & losers for mover chips
    sorted_up = valid.nlargest(3, "Day Change %")
    sorted_down = valid.nsmallest(3, "Day Change %")

    gainer_chips = "".join(
        f'<span class="mover-chip gainer">{r["Ticker"]} ▲{r["Day Change %"]:.2f}%</span>'
        for _, r in sorted_up.iterrows() if r["Day Change %"] > 0
    )
    loser_chips = "".join(
        f'<span class="mover-chip loser">{r["Ticker"]} ▼{abs(r["Day Change %"]):.2f}%</span>'
        for _, r in sorted_down.iterrows() if r["Day Change %"] < 0
    )

    report_date = meta.get("report_date", "—")
    num_positions = meta.get("total_positions", len(holdings))
    refresh_time = now_et.strftime("%I:%M %p ET")

    st.markdown(
        f"""
        <div class="title-bar">
            <h1>Portfolio Holdings</h1>
            <div class="title-meta">
                <span class="meta-chip">
                    <span class="meta-label">As of</span>
                    <span class="meta-icon">📅</span> {report_date}
                </span>
                <span class="meta-chip">
                    <span class="meta-label">Positions</span>
                    <span class="meta-icon">📊</span> {num_positions}
                </span>
                <span class="meta-chip">
                    <span class="meta-label">FX</span>
                    USD/CAD {fx_rate:.4f}
                </span>
                <span class="meta-chip">
                    <span style="color:#34d399;">{gainers} ▲</span>
                    &nbsp;/&nbsp;
                    <span style="color:#f87171;">{losers} ▼</span>
                </span>
                <span class="market-status">
                    <span class="dot {market_dot}"></span> {market_label}
                </span>
            </div>
            <div class="movers-strip">
                {gainer_chips}{loser_chips}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 1 — HOLDINGS / MODEL TOGGLE
    # ══════════════════════════════════════════════════════════════════════

    model_meta, model_df = load_model_portfolio()
    has_model = not model_df.empty

    if has_model:
        view = st.segmented_control(
            "View",
            options=["Holdings", "Model Portfolio"],
            default="Holdings",
            label_visibility="collapsed",
        )
    else:
        view = "Holdings"

    if view == "Holdings":
        # ── Holdings view (unchanged) ──
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
                return "color: #475569"
            return "color: #34d399" if val >= 0 else "color: #f87171"

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

    else:
        # ── Model Portfolio comparison view ──
        st.markdown('<div class="section-header">Model Portfolio Comparison</div>',
                    unsafe_allow_html=True)
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
                key="model_ticker_filter",
            )
        with m_col2:
            m_sel_status = st.multiselect(
                "Filter by status",
                sorted(comparison["Status"].unique()),
                placeholder="Filter by status...",
                key="model_status_filter",
            )

        display_model = comparison.copy()
        if m_sel_tickers:
            display_model = display_model[display_model["Ticker"].isin(m_sel_tickers)]
        if m_sel_status:
            display_model = display_model[display_model["Status"].isin(m_sel_status)]

        def color_divergence(val):
            if pd.isna(val) or val == 0:
                return "color: #64748b"
            return "color: #34d399" if val > 0 else "color: #f87171"

        def color_status(val):
            if val == "Model Only":
                return "color: #f87171"
            if val == "Not in Model":
                return "color: #f59e0b"
            return "color: #94a3b8"

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

        # Summary metrics
        in_both = len(comparison[comparison["Status"] == "Matched"])
        model_only = len(comparison[comparison["Status"] == "Model Only"])
        not_in_model = len(comparison[comparison["Status"] == "Not in Model"])
        avg_div = comparison["Divergence %"].abs().mean()

        st.markdown(
            f"""
            <div class="metric-row">
                <div class="metric-card">
                    <div class="label">Matched Positions</div>
                    <div class="value">{in_both}</div>
                    <div class="delta" style="color:#64748b; font-size:0.78rem;">
                        In both model & portfolio
                    </div>
                </div>
                <div class="metric-card">
                    <div class="label">Model Only</div>
                    <div class="value" style="color:#f87171;">{model_only}</div>
                    <div class="delta" style="color:#64748b; font-size:0.78rem;">
                        In model but not held
                    </div>
                </div>
                <div class="metric-card">
                    <div class="label">Not in Model</div>
                    <div class="value" style="color:#f59e0b;">{not_in_model}</div>
                    <div class="delta" style="color:#64748b; font-size:0.78rem;">
                        Held but not in model
                    </div>
                </div>
                <div class="metric-card">
                    <div class="label">Avg. Abs. Divergence</div>
                    <div class="value">{avg_div:.2f}%</div>
                    <div class="delta" style="color:#64748b; font-size:0.78rem;">
                        Mean |actual - target|
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
            "CURRENCY EXPOSURE", ["#34d399", "#f87171"],
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
                    <div class="value" style="font-size:1.6rem;">
                        {f'{w_pe:.1f}x' if w_pe else '—'}
                    </div>
                    <div class="delta" style="color:#64748b; font-size:0.78rem;">
                        {f'Forward: {w_fpe:.1f}x' if w_fpe else 'Forward: —'}
                        &nbsp;&middot;&nbsp; Coverage: {pe_cov:.0f}%
                    </div>
                </div>
                <div class="metric-card">
                    <div class="label">Weighted Dividend Yield</div>
                    <div class="value" style="font-size:1.6rem; color:#34d399;">
                        {f'{w_yield*100:.2f}%' if w_yield else '—'}
                    </div>
                    <div class="delta" style="color:#64748b; font-size:0.78rem;">
                        {f'~${metrics["total_value"]*w_yield/1e6:.1f}M annual income' if w_yield else ''}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="label">Weighted Beta</div>
                    <div class="value" style="font-size:1.6rem;">
                        {f'{w_beta:.2f}' if w_beta else '—'}
                    </div>
                    <div class="delta" style="color:{'#34d399' if w_beta and w_beta < 1 else '#f87171' if w_beta and w_beta > 1 else '#64748b'}; font-size:0.78rem;">
                        {'Lower volatility than market' if w_beta and w_beta < 1 else 'Higher volatility than market' if w_beta and w_beta > 1 else ''}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="label">52-Week Positioning</div>
                    <div class="value" style="font-size:1.6rem;">
                        {f'{w_52w:.0f}%' if w_52w is not None else '—'}
                    </div>
                    <div class="delta" style="color:#64748b; font-size:0.78rem;">
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
    # SECTION 3 — MARKET INTEL
    # ══════════════════════════════════════════════════════════════════════

    st.markdown('<div class="section-header">Latest Market Intel</div>',
                unsafe_allow_html=True)
    st.caption("Top stories from today's biggest movers · Refreshes hourly")

    with st.spinner("Scanning news for biggest movers..."):
        top_stories = get_top_movers_news(table, holdings)

    if not top_stories:
        st.info("No relevant news found for today's movers.")
    else:
        for item in top_stories:
            arrow, sent_color = SENTIMENT_STYLE.get(item["sentiment"], ("—", "#64748b"))

            chg = item["day_change"]
            chg_color = "#34d399" if chg >= 0 else "#f87171"
            chg_arrow = "▲" if chg >= 0 else "▼"

            st.markdown(
                f"""
                <div class="news-card">
                    <div class="news-top">
                        <span class="ticker-badge"
                              style="background:{chg_color}18; color:{chg_color}; border:1px solid {chg_color}44;">
                            {item["ticker"]} {chg_arrow} {abs(chg):.2f}%
                        </span>
                        <span class="sentiment-badge" style="color:{sent_color};">
                            {arrow} {item["sentiment"]}
                        </span>
                        <span class="publisher">{item["publisher"]}</span>
                    </div>
                    <div class="headline">
                        <a href="{item["link"]}" target="_blank">{item["title"]}</a>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Footer ──
    st.markdown(
        f'<div class="footer-text">'
        f'Last refreshed: {refresh_time} &middot; '
        f'Report: {meta.get("report_date", "—")} &middot; '
        f'Live prices via Yahoo Finance &middot; '
        f'Sentiment via TextBlob'
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
