"""
================================================================================
 DATA IMPORT PAGE
 ────────────────
 Upload holdings (CSV/Excel) and edit model portfolios in-app.
================================================================================
"""

import io
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
HOLDINGS_FILE = BASE_DIR / "holdings.json"
MODELS_DIR = BASE_DIR / "models"

SECTOR_OPTIONS = [
    "Financials", "Technology", "Energy", "Utilities", "Industrials",
    "Healthcare", "Consumer Staples", "Consumer Discretionary",
    "Real Estate", "Telecommunications", "Commodities",
    "Digital Assets", "ETF - Equity", "Other",
]

# Ticker suffix mapping: broker suffix → yfinance suffix
TICKER_SUFFIX_MAP = {
    "-T":  ".TO",   # TSX
    "-N":  "",      # NYSE
    "-O":  "",      # NASDAQ
    "-P":  "",      # NYSE Arca / other
    "-US": "",      # US OTC
    "-5":  "",      # Composite / OTC (e.g. NSRGY-5, LVMUY-5)
    "-I":  "",      # US exchange (e.g. ITA-I)
    "-GD": "",      # Globally diversified / misc ETFs
    "-V":  ".V",    # TSX Venture
    "-CT": ".TO",   # TSX (alternate)
    "-CF": ".TO",   # TSX (alternate)
    "-CN": ".CN",   # CSE
}

# Broker section name → dashboard sector
SECTION_SECTOR_MAP = {
    "BASIC MATERIALS":      "Commodities",
    "ENERGY":               "Energy",
    "INDUSTRIAL":           "Industrials",
    "CONSUMER,  CYCLICAL":  "Consumer Discretionary",
    "CONSUMER,  NON-CYCLICAL": "Consumer Staples",
    "FINANCIAL":            "Financials",
    "TECHNOLOGY":           "Technology",
    "COMMUNICATIONS":       "Telecommunications",
    "UTILITIES":            "Utilities",
    "EQUITY FUNDS & EXCHANGE TRADED FUNDS (ETFS)": "ETF - Equity",
    "BOND FUNDS & EXCHANGE TRADED FUNDS (ETFS)": "Other",
    "OTHER":                "Other",
    "RATE RESET PREFERREDS": "Financials",
    "CONVERTIBLE DEBENTURES": "Other",
    "ALTERNATIVE INVESTMENTS": "Other",
    "MULTI-ASSET":          "Other",
}

# ──────────────────────────────────────────────────────────────────────────────
# CSS
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

.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

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

.status-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s ease;
}
.status-card:hover {
    border-color: var(--border-hover);
}
.status-card .label {
    font-size: 0.65rem;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.status-card .value {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-top: 0.25rem;
}
.status-card .detail {
    font-size: 0.75rem;
    color: var(--text-secondary);
    margin-top: 0.25rem;
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
}

hr {
    border: none;
    border-top: 1px solid var(--border-subtle);
    margin: 2rem 0;
}

/* ── Data table ── */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    overflow: hidden;
}

/* ── File uploader styling ── */
div[data-testid="stFileUploader"] > div {
    border-radius: var(--radius-md);
}

/* ── Custom scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-default); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--border-hover); }
</style>
"""


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────


def load_current_holdings() -> dict | None:
    """Load current holdings.json."""
    if not HOLDINGS_FILE.exists():
        return None
    try:
        with open(HOLDINGS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return None


def load_current_models() -> list[dict]:
    """Load metadata from all model JSON files."""
    models = []
    if not MODELS_DIR.exists():
        return models
    for fpath in sorted(MODELS_DIR.glob("*.json")):
        try:
            with open(fpath, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            meta = data.get("_meta", {})
            meta["_file"] = fpath.name
            meta["_path"] = str(fpath)
            meta["_num"] = len(data.get("constituents", []))
            models.append(meta)
        except Exception:
            pass
    return models


def load_model_data(filepath: str) -> tuple[dict, list[dict]]:
    """Load full model data (meta + constituents) from a JSON file."""
    with open(filepath, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return data.get("_meta", {}), data.get("constituents", [])


def enrich_row(ticker: str) -> dict:
    """Enrich a single ticker with sector/currency from holdings."""
    hdata = load_current_holdings()
    if hdata:
        for h in hdata.get("holdings", []):
            if h["ticker"] == ticker:
                return {"sector": h.get("sector", "Other"),
                        "currency": h.get("currency", "USD")}
    # Fallback
    return {
        "sector": "Other",
        "currency": "CAD" if ticker.endswith(".TO") else "USD",
    }


def consolidate_holdings(holdings: list[dict]) -> list[dict]:
    """Consolidate same-ticker rows (long + short netting).

    Sums quantity and market_value for identical tickers.
    Cross-listed pairs (X.TO ↔ X) are kept separate to preserve
    correct currency exposure and sector allocations.
    """
    from collections import OrderedDict

    by_ticker: OrderedDict[str, dict] = OrderedDict()
    for h in holdings:
        t = h["ticker"]
        if t in by_ticker:
            by_ticker[t]["quantity"] += h["quantity"]
            by_ticker[t]["market_value"] += h["market_value"]
        else:
            by_ticker[t] = dict(h)  # copy

    # Drop zero/tiny positions after netting
    result = []
    for h in by_ticker.values():
        if abs(h["market_value"]) < 1:
            continue
        h["market_value"] = round(h["market_value"])
        q = h["quantity"]
        h["quantity"] = round(q, 4) if q != int(q) else int(q)
        result.append(h)

    return result


def _has_exchange_suffix(raw: str) -> bool:
    """Check if a broker symbol has a recognized exchange suffix."""
    return any(raw.endswith(sfx) for sfx in TICKER_SUFFIX_MAP)


def convert_broker_ticker(raw: str) -> str:
    """Convert broker symbol to yfinance ticker.

    Examples:  GOOGL-O → GOOGL,  BMO-T → BMO.TO,  V-N → V,  ZQQ-T → ZQQ.TO
    Handles:  BGU'U-T → BGU-U.TO,  BIP.UN-T → BIP-UN.TO
    """
    raw = raw.strip()
    for suffix, replacement in TICKER_SUFFIX_MAP.items():
        if raw.endswith(suffix):
            base = raw[: -len(suffix)]
            # Replace apostrophes and dots with hyphens for yfinance
            # e.g. BGU'U → BGU-U,  BIP.UN → BIP-UN
            base = base.replace("'", "-").replace(".", "-")
            return base + replacement
    return raw


def _find_header_row(df: pd.DataFrame) -> int | None:
    """Find the row containing column headers like 'Symbol' or 'Quantity'."""
    for i in range(min(10, len(df))):
        row_vals = [str(v).strip().lower() for v in df.iloc[i]]
        if "symbol" in row_vals or "ticker" in row_vals:
            return i
    return None


def parse_holdings_file(file_bytes: bytes, filename: str) -> tuple[list[dict], str] | None:
    """Parse a holdings file (CSV or Excel) from broker export.

    Handles the Scotia / broker XLS format with:
    - Date header row above column headers
    - Section headers (EQUITY, ENERGY, FINANCIAL, etc.) for sector inference
    - "Total" summary rows to skip
    - Broker ticker format (GOOGL-O, BMO-T, etc.)

    Also handles simple flat CSV/Excel with standard column names.

    Returns (holdings_list, report_date_str) or None if parsing fails.
    """
    # ── Read raw DataFrame (no header) ──
    try:
        if filename.lower().endswith((".xlsx", ".xls")):
            df_raw = pd.read_excel(io.BytesIO(file_bytes), header=None)
        else:
            text = file_bytes.decode("utf-8-sig")
            df_raw = pd.read_csv(io.StringIO(text), header=None)
    except Exception:
        return None

    # ── Detect header row ──
    header_idx = _find_header_row(df_raw)
    if header_idx is None:
        # Fall back: maybe headers are in row 0 (simple flat file)
        header_idx = 0

    # Extract report date from rows above the header (if present)
    report_date = datetime.now().strftime("%Y-%m-%d")
    for i in range(header_idx):
        cell = str(df_raw.iloc[i, 0]).strip()
        if cell.lower().startswith("as of "):
            # "As of February 27, 2026" → parse date
            try:
                report_date = datetime.strptime(
                    cell.replace("As of ", "").strip(), "%B %d, %Y"
                ).strftime("%Y-%m-%d")
            except ValueError:
                pass

    # Set column names from the header row
    headers = [str(v).strip() for v in df_raw.iloc[header_idx]]
    df_raw.columns = headers
    df = df_raw.iloc[header_idx + 1:].reset_index(drop=True)

    # ── Build column index (flexible matching) ──
    col_map: dict[str, str] = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl in ("security description", "name", "security name",
                   "security", "description"):
            col_map["name"] = col
        elif cl in ("symbol", "ticker", "ticker symbol"):
            col_map["ticker"] = col
        elif cl in ("security currency", "currency", "curr", "ccy"):
            col_map["currency"] = col
        elif cl in ("quantity", "shares", "units", "qty"):
            col_map["quantity"] = col
        elif cl in ("market value", "value", "mkt value", "market_value",
                     "mktval", "market val"):
            col_map["market_value"] = col
        elif cl in ("average cost", "avg cost", "avg_cost", "cost"):
            col_map["avg_cost"] = col
        elif cl in ("book value", "book_value", "book val"):
            col_map["book_value"] = col
        elif cl in ("price",):
            col_map["price"] = col
        elif cl in ("% of portfolio", "pct_of_portfolio", "weight", "weight %"):
            col_map["pct"] = col
        elif cl in ("estimated annual income", "est annual income",
                     "annual income", "income"):
            col_map["est_income"] = col
        elif cl in ("yield (%)", "yield", "yield_pct"):
            col_map["yield"] = col
        elif cl in ("sector", "asset class", "gics sector"):
            col_map["sector"] = col

    # Must have at least ticker/symbol and market value
    if "ticker" not in col_map or "market_value" not in col_map:
        return None

    # ── Build existing sector lookup from current holdings.json ──
    existing_sectors: dict[str, str] = {}
    existing_data = load_current_holdings()
    if existing_data:
        for h in existing_data.get("holdings", []):
            existing_sectors[h["ticker"]] = h.get("sector", "Other")

    # ── Parse rows, tracking current section for fallback sector ──
    holdings = []
    current_section = "Other"

    for _, row in df.iterrows():
        ticker_raw = str(row.get(col_map["ticker"], "")).strip()
        desc = str(row.get(col_map.get("name", ""), "")).strip()

        # Skip empty rows
        if (not ticker_raw or ticker_raw == "nan") and (not desc or desc == "nan"):
            continue

        # Section header: no ticker, desc matches a known section
        if (not ticker_raw or ticker_raw == "nan"):
            # Check if it's a section header (not a "Total" row)
            if not desc.startswith("Total") and desc != "nan":
                if desc in SECTION_SECTOR_MAP:
                    current_section = SECTION_SECTOR_MAP[desc]
                elif desc.upper() == desc and len(desc) > 2:
                    # All-caps section we don't have a mapping for
                    current_section = "Other"
            continue

        # Skip "Total" rows that somehow have a symbol
        if desc.startswith("Total"):
            continue

        # Skip cash-like rows
        if ticker_raw.startswith("CASH-") or ticker_raw.startswith("DYN"):
            continue

        # Only include exchange-traded securities (skip CUSIPs, GIC codes, etc.)
        # Exchange tickers end with -T, -N, -O, -P, -US, etc.
        if not _has_exchange_suffix(ticker_raw):
            continue

        # ── Parse data row ──
        try:
            mkt_val_str = str(row.get(col_map["market_value"], "0"))
            mkt_val_str = mkt_val_str.replace(",", "").replace("$", "").replace("*", "")
            market_value = float(mkt_val_str)
            if market_value == 0:
                continue

            qty_str = str(row.get(col_map["quantity"], "0"))
            qty_str = qty_str.replace(",", "").replace("*", "")
            quantity = float(qty_str)

            # Convert broker ticker → yfinance ticker
            ticker = convert_broker_ticker(ticker_raw)

            # Currency
            currency = "USD"
            if "currency" in col_map:
                curr = str(row.get(col_map["currency"], "")).strip().upper()
                if curr in ("CAD", "USD", "EUR"):
                    currency = curr

            # Sector: prefer existing holdings.json designation,
            # fall back to broker section mapping for new tickers
            sector = existing_sectors.get(ticker, current_section)

            # Clean name
            name = desc if desc and desc != "nan" else ticker

            holdings.append({
                "name": name,
                "ticker": ticker,
                "currency": currency,
                "quantity": round(quantity, 4) if quantity != int(quantity) else int(quantity),
                "market_value": round(market_value),
                "sector": sector,
            })
        except (ValueError, TypeError, KeyError):
            continue

    if not holdings:
        return None

    # Consolidate: merge same-ticker rows and cross-listed pairs
    holdings = consolidate_holdings(holdings)

    return (holdings, report_date)


# ──────────────────────────────────────────────────────────────────────────────
# STREAMLIT PAGE
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Data Import", page_icon="📤", layout="wide")
st.markdown(PAGE_CSS, unsafe_allow_html=True)

st.markdown("# 📤 Data Import")
st.caption("Upload holdings and edit model portfolios to update your dashboard.")

# ── Current data status ──
hdata = load_current_holdings()
models_info = load_current_models()

col1, col2 = st.columns(2)

with col1:
    if hdata:
        meta = hdata.get("_meta", {})
        st.markdown(
            f"""
            <div class="status-card">
                <div class="label">Current Holdings</div>
                <div class="value">{meta.get('total_positions', '?')} positions</div>
                <div class="detail">
                    Last updated: {meta.get('report_date', 'Unknown')} ·
                    ${meta.get('total_market_value', 0):,.0f} total
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("No holdings.json found")

with col2:
    if models_info:
        model_summary = " · ".join(
            f"{m.get('tab_name', m.get('model_name', '?'))} ({m['_num']})"
            for m in models_info
        )
        st.markdown(
            f"""
            <div class="status-card">
                <div class="label">Current Models</div>
                <div class="value">{len(models_info)} models</div>
                <div class="detail">{model_summary}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("No model files found in models/")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — HOLDINGS IMPORT
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">Upload Holdings</div>',
            unsafe_allow_html=True)

holdings_file = st.file_uploader(
    "Drag & drop your broker holdings export (CSV or Excel)",
    type=["csv", "xlsx", "xls"],
    key="holdings_upload",
    help="Columns: Name, Ticker, Currency, Quantity, Market Value, Sector",
)

if holdings_file is not None:
    raw_bytes = holdings_file.read()
    result = parse_holdings_file(raw_bytes, holdings_file.name)

    if result is None:
        st.error(
            "Could not parse holdings file. Please ensure it has columns like: "
            "**Symbol**, **Market Value**, **Quantity**."
        )
        # Show raw preview so user can see what we received
        st.caption("Raw file preview:")
        try:
            if holdings_file.name.lower().endswith((".xlsx", ".xls")):
                preview_df = pd.read_excel(io.BytesIO(raw_bytes))
            else:
                preview_df = pd.read_csv(
                    io.BytesIO(raw_bytes), encoding="utf-8-sig")
            st.dataframe(preview_df.head(10), use_container_width=True)
            st.caption(
                f"Detected columns: {', '.join(str(c) for c in preview_df.columns.tolist())}")
        except Exception:
            pass
    else:
        parsed, report_date = result

        # Show preview
        preview_df = pd.DataFrame(parsed)
        total_value = preview_df["market_value"].sum()
        num_sectors = preview_df["sector"].nunique()

        st.success(
            f"Parsed **{len(parsed)} positions** · "
            f"**${total_value:,.0f}** total · "
            f"**{num_sectors} sectors** · "
            f"Report date: **{report_date}**"
        )

        st.dataframe(
            preview_df[["ticker", "name", "currency", "quantity",
                         "market_value", "sector"]],
            use_container_width=True,
            height=300,
            hide_index=True,
        )

        # Confirm import
        if st.button("Confirm Holdings Import", type="primary",
                      key="confirm_holdings"):
            output = {
                "_meta": {
                    "report_date": report_date,
                    "report_title": f"Holdings Import — {report_date}",
                    "total_positions": len(parsed),
                    "total_market_value": round(total_value),
                    "notes": f"Imported from {holdings_file.name}",
                },
                "holdings": parsed,
            }
            with open(HOLDINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2)
            st.cache_data.clear()
            st.success(f"Saved {len(parsed)} holdings to holdings.json")
            st.balloons()

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MODEL EDITOR
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">Edit Model Portfolio</div>',
            unsafe_allow_html=True)

# ── Model selector ──
model_options = ["➕ Create New Model"] + [
    m.get("tab_name", m.get("model_name", m["_file"]))
    for m in models_info
]

selected_model = st.selectbox(
    "Select model to edit",
    options=model_options,
    key="model_selector",
)

is_new = selected_model == "➕ Create New Model"

# ── Load existing model or start fresh ──
if is_new:
    editor_df = pd.DataFrame(columns=[
        "ticker", "name", "target_pct", "sector", "currency",
    ])
    current_tab_name = ""
    current_cash = 0.0
    current_model_name = ""
    current_file = None
else:
    # Find the matching model
    model_meta = None
    for m in models_info:
        if m.get("tab_name", m.get("model_name", m["_file"])) == selected_model:
            model_meta = m
            break

    if model_meta:
        meta, constituents = load_model_data(model_meta["_path"])
        editor_df = pd.DataFrame(constituents)
        # Ensure required columns exist
        for col in ["ticker", "name", "target_pct", "sector", "currency"]:
            if col not in editor_df.columns:
                editor_df[col] = ""
        editor_df = editor_df[["ticker", "name", "target_pct", "sector", "currency"]]
        current_tab_name = meta.get("tab_name", "")
        current_cash = meta.get("cash_pct", 0.0)
        current_model_name = meta.get("model_name", "")
        current_file = model_meta["_file"]
    else:
        editor_df = pd.DataFrame(columns=[
            "ticker", "name", "target_pct", "sector", "currency",
        ])
        current_tab_name = ""
        current_cash = 0.0
        current_model_name = ""
        current_file = None

# ── Tab name and cash % ──
name_col, cash_col = st.columns([2, 1])

with name_col:
    tab_name = st.text_input(
        "Model tab name",
        value=current_tab_name,
        placeholder="e.g. NA Div Model",
        help="Short name shown as a tab on the dashboard",
        key="editor_tab_name",
    )

with cash_col:
    cash_pct = st.number_input(
        "Cash %",
        value=current_cash,
        min_value=0.0,
        max_value=100.0,
        step=0.25,
        format="%.2f",
        key="editor_cash_pct",
    )

# ── Editable table ──
edited_df = st.data_editor(
    editor_df,
    num_rows="dynamic",
    use_container_width=True,
    height=450,
    key="model_editor",
    column_config={
        "ticker": st.column_config.TextColumn(
            "Ticker",
            help="yfinance ticker (e.g. GOOGL, RY.TO)",
            width="small",
        ),
        "name": st.column_config.TextColumn(
            "Name",
            help="Security name",
            width="medium",
        ),
        "target_pct": st.column_config.NumberColumn(
            "Target %",
            help="Target allocation percentage",
            min_value=0.0,
            max_value=100.0,
            step=0.05,
            format="%.2f",
            width="small",
        ),
        "sector": st.column_config.SelectboxColumn(
            "Sector",
            options=SECTOR_OPTIONS,
            width="medium",
        ),
        "currency": st.column_config.SelectboxColumn(
            "Currency",
            options=["CAD", "USD"],
            width="small",
        ),
    },
)

# ── Summary stats ──
valid_rows = edited_df.dropna(subset=["ticker"])
valid_rows = valid_rows[valid_rows["ticker"].str.strip() != ""]
total_equity = valid_rows["target_pct"].sum() if not valid_rows.empty else 0.0
num_constituents = len(valid_rows)

st.caption(
    f"**{num_constituents} constituents** · "
    f"**{total_equity:.2f}% equity** · "
    f"**{cash_pct:.2f}% cash** · "
    f"**{total_equity + cash_pct:.2f}% total**"
)

if abs(total_equity + cash_pct - 100.0) > 0.5:
    st.warning(
        f"Total allocation is {total_equity + cash_pct:.2f}% "
        f"(expected ~100%). Adjust target percentages or cash."
    )

# ── Save / Delete buttons ──
btn_col1, btn_col2, btn_spacer = st.columns([1, 1, 2])

with btn_col1:
    save_clicked = st.button(
        "Save Model", type="primary", key="save_model",
        disabled=(num_constituents == 0),
    )

with btn_col2:
    if not is_new and current_file:
        delete_clicked = st.button(
            "Delete Model", type="secondary", key="delete_model",
        )
    else:
        delete_clicked = False

# ── Save logic ──
if save_clicked:
    effective_tab = tab_name.strip() if tab_name and tab_name.strip() else ""
    if not effective_tab:
        st.error("Please enter a model tab name.")
    elif num_constituents == 0:
        st.error("Add at least one constituent.")
    else:
        # Build constituents list, auto-enrich missing sector/currency
        constituents = []
        for _, row in valid_rows.iterrows():
            ticker = str(row["ticker"]).strip()
            name = str(row.get("name", "")).strip()
            target = float(row.get("target_pct", 0))
            sector = str(row.get("sector", "")).strip()
            currency = str(row.get("currency", "")).strip()

            # Auto-enrich if sector/currency missing
            if not sector or sector in ("", "nan", "None"):
                enriched = enrich_row(ticker)
                sector = enriched["sector"]
            if not currency or currency in ("", "nan", "None"):
                enriched = enrich_row(ticker)
                currency = enriched["currency"]

            constituents.append({
                "name": name,
                "ticker": ticker,
                "target_pct": round(target, 2),
                "sector": sector,
                "currency": currency,
            })

        slug = re.sub(r"[^a-z0-9]+", "_", effective_tab.lower()).strip("_")
        out_path = MODELS_DIR / f"{slug}.json"
        MODELS_DIR.mkdir(exist_ok=True)

        output = {
            "_meta": {
                "last_updated": datetime.now().strftime("%Y-%m-%d"),
                "source": "Edited in dashboard",
                "model_name": current_model_name or effective_tab,
                "tab_name": effective_tab,
                "total_equity_pct": round(total_equity, 2),
                "cash_pct": round(cash_pct, 2),
                "num_constituents": num_constituents,
            },
            "constituents": constituents,
        }

        # If renaming, delete old file
        if current_file and current_file != f"{slug}.json":
            old_path = MODELS_DIR / current_file
            if old_path.exists():
                old_path.unlink()

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        st.cache_data.clear()
        st.success(
            f"Saved **{effective_tab}** — "
            f"{num_constituents} constituents to `{out_path.name}`"
        )
        st.balloons()

# ── Delete logic ──
if delete_clicked and current_file:
    del_path = MODELS_DIR / current_file
    if del_path.exists():
        del_path.unlink()
        st.cache_data.clear()
        st.success(f"Deleted `{current_file}`")
        st.rerun()
