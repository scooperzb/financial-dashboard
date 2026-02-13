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
    display: flex;
    align-items: baseline;
    gap: 1rem;
    margin-bottom: 0.25rem;
}
.title-bar h1 {
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0;
    color: #f1f5f9;
}
.title-bar .subtitle {
    font-size: 0.85rem;
    color: #64748b;
    font-weight: 400;
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
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.6,
        marker=dict(colors=colors, line=dict(color="#0f172a", width=2)) if colors
               else dict(line=dict(color="#0f172a", width=2)),
        textinfo="label+percent", textposition="outside",
        textfont=dict(size=11, color="#94a3b8"),
        hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>",
        sort=True, direction="clockwise",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#94a3b8"),
                   x=0.5, xanchor="center"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=380, margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
        annotations=[dict(
            text=f"<b>${sum(values)/1e6:.0f}M</b>",
            x=0.5, y=0.5, font=dict(size=18, color="#e2e8f0"), showarrow=False,
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

    st.markdown(
        f'<div class="title-bar">'
        f'<h1>Portfolio Holdings</h1>'
        f'<span class="subtitle">{meta.get("report_date", "—")} &middot; '
        f'{meta.get("total_positions", len(holdings))} positions</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="label">Gainers / Losers</div>
                <div class="value" style="font-size:1.4rem;">
                    <span style="color:#34d399;">{gainers} ▲</span>
                    &nbsp;/&nbsp;
                    <span style="color:#f87171;">{losers} ▼</span>
                </div>
            </div>
            <div class="metric-card">
                <div class="label">USD / CAD</div>
                <div class="value" style="font-size:1.4rem;">{fx_rate:.4f}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 1 — HOLDINGS TABLE
    # ══════════════════════════════════════════════════════════════════════

    st.markdown('<div class="section-header">Holdings</div>', unsafe_allow_html=True)

    display = table[["Ticker", "Shares", "Price", "Day Change %",
                      "Value (CAD)", "Sector"]].copy()

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

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 2 — DONUT CHARTS
    # ══════════════════════════════════════════════════════════════════════

    st.markdown('<div class="section-header">Portfolio Composition</div>',
                unsafe_allow_html=True)

    chart_left, chart_right = st.columns(2)

    with chart_left:
        merged = table.copy()
        merged["Currency"] = holdings["currency"].values
        curr_agg = merged.groupby("Currency")["Value (CAD)"].sum()
        fig_curr = make_donut(
            curr_agg.index.tolist(), curr_agg.values.tolist(),
            "CURRENCY EXPOSURE", ["#34d399", "#f87171"],
        )
        st.plotly_chart(fig_curr, use_container_width=True)

    with chart_right:
        merged["Sector"] = holdings["sector"].values
        sect_agg = merged.groupby("Sector")["Value (CAD)"].sum().sort_values(ascending=False)
        sector_colors = [SECTOR_COLORS.get(s, "#6b7280") for s in sect_agg.index]
        fig_sect = make_donut(
            sect_agg.index.tolist(), sect_agg.values.tolist(),
            "SECTOR ALLOCATION", sector_colors,
        )
        st.plotly_chart(fig_sect, use_container_width=True)

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
        f'Report: {meta.get("report_date", "—")} &middot; '
        f'Live prices via Yahoo Finance &middot; '
        f'Sentiment via TextBlob &middot; '
        f'Built with Streamlit + Plotly'
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
