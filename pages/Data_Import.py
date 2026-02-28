"""
================================================================================
 DATA IMPORT PAGE
 ────────────────
 Upload holdings CSVs and model portfolio CSVs to update the dashboard data.
 Provides preview, validation, and one-click import.
================================================================================
"""

import csv
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

# Ticker suffix mapping (same as convert_model.py)
SUFFIX_MAP = {
    "-T": ".TO",   # TSX
    "-N": "",      # NYSE
    "-O": "",      # NASDAQ
}

# ──────────────────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────────────────

PAGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

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

.status-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 1rem;
}
.status-card .label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.status-card .value {
    font-size: 1.3rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-top: 0.2rem;
}
.status-card .detail {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 0.2rem;
}
</style>
"""


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────


def convert_ticker(raw: str) -> str:
    """Convert 'ARE-T' -> 'ARE.TO', 'GOOGL-O' -> 'GOOGL', etc."""
    raw = raw.strip()
    for suffix, replacement in SUFFIX_MAP.items():
        if raw.endswith(suffix):
            base = raw[: -len(suffix)]
            return base + replacement
    return raw


def load_current_holdings() -> dict | None:
    """Load current holdings.json metadata."""
    if not HOLDINGS_FILE.exists():
        return None
    try:
        with open(HOLDINGS_FILE, "r") as f:
            data = json.load(f)
        return data
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
            meta["_num"] = len(data.get("constituents", []))
            models.append(meta)
        except Exception:
            pass
    return models


def enrich_constituents(constituents: list[dict]) -> list[dict]:
    """Add sector and currency to model constituents using holdings lookup."""
    # Build lookup from holdings
    holdings_lookup = {}
    hdata = load_current_holdings()
    if hdata:
        for h in hdata.get("holdings", []):
            holdings_lookup[h["ticker"]] = {
                "sector": h.get("sector", "Other"),
                "currency": h.get("currency", "USD"),
            }

    for c in constituents:
        ticker = c["ticker"]
        # Tier 1: from holdings
        if ticker in holdings_lookup:
            c["sector"] = holdings_lookup[ticker]["sector"]
            c["currency"] = holdings_lookup[ticker]["currency"]
        else:
            # Tier 2: infer
            c["currency"] = "CAD" if ticker.endswith(".TO") else "USD"
            c["sector"] = "Other"

    return constituents


def parse_model_csv(file_bytes: bytes) -> tuple[list[dict], float, str]:
    """Parse a model CSV and return (constituents, cash_pct, model_name).

    Reuses logic from convert_model.py.
    """
    text = file_bytes.decode("utf-8-sig")
    reader = csv.reader(io.StringIO(text))

    constituents = []
    cash_pct = 0.0
    model_name = ""

    for row in reader:
        if not row:
            continue

        # Model name from first non-header data row
        if row[0].strip() and row[0].strip() != "Name" and not model_name:
            model_name = row[0].strip()

        # Cash row
        if len(row) >= 5 and row[1].strip() == "Currency/Cash":
            try:
                cash_pct = float(row[4].strip())
            except (ValueError, IndexError):
                pass
            continue

        # Equity rows
        if len(row) < 5 or not row[3].strip():
            continue
        raw_ticker = row[3].strip()
        if raw_ticker == "Ticker":
            continue

        name = row[2].strip()
        try:
            tgt_pct = float(row[4].strip())
        except (ValueError, IndexError):
            continue
        if tgt_pct == 0:
            continue

        yf_ticker = convert_ticker(raw_ticker)
        exchange = (
            "TSX" if raw_ticker.endswith("-T")
            else "NASDAQ" if raw_ticker.endswith("-O")
            else "NYSE"
        )

        constituents.append({
            "name": name,
            "ticker": yf_ticker,
            "raw_ticker": raw_ticker,
            "exchange": exchange,
            "target_pct": tgt_pct,
        })

    return constituents, cash_pct, model_name


def parse_holdings_csv(file_bytes: bytes) -> list[dict] | None:
    """Parse a holdings CSV from broker export.

    Expected columns (flexible matching):
      - Name / Security Name
      - Ticker / Symbol
      - Currency / Curr
      - Quantity / Shares / Units
      - Market Value / Value
      - Sector / Asset Class

    Returns list of holdings dicts or None if parsing fails.
    """
    text = file_bytes.decode("utf-8-sig")
    df = pd.read_csv(io.StringIO(text))

    # Normalize column names for flexible matching
    col_map = {}
    for col in df.columns:
        cl = col.strip().lower()
        if cl in ("name", "security name", "security", "description"):
            col_map["name"] = col
        elif cl in ("ticker", "symbol", "ticker symbol"):
            col_map["ticker"] = col
        elif cl in ("currency", "curr", "ccy"):
            col_map["currency"] = col
        elif cl in ("quantity", "shares", "units", "qty"):
            col_map["quantity"] = col
        elif cl in ("market value", "value", "mkt value", "market_value", "mktval"):
            col_map["market_value"] = col
        elif cl in ("sector", "asset class", "gics sector"):
            col_map["sector"] = col

    # Check required columns
    required = ["name", "ticker", "quantity", "market_value"]
    missing = [k for k in required if k not in col_map]
    if missing:
        return None

    holdings = []
    for _, row in df.iterrows():
        try:
            ticker = str(row[col_map["ticker"]]).strip()
            if not ticker or ticker == "nan":
                continue

            name = str(row[col_map["name"]]).strip()
            quantity = float(str(row[col_map["quantity"]]).replace(",", ""))
            market_value = float(str(row[col_map["market_value"]]).replace(",", "").replace("$", ""))

            currency = "USD"
            if "currency" in col_map:
                curr = str(row[col_map["currency"]]).strip().upper()
                if curr in ("CAD", "USD"):
                    currency = curr

            sector = "Other"
            if "sector" in col_map:
                s = str(row[col_map["sector"]]).strip()
                if s and s != "nan":
                    sector = s

            holdings.append({
                "name": name,
                "ticker": ticker,
                "currency": currency,
                "quantity": int(quantity),
                "market_value": round(market_value),
                "sector": sector,
            })
        except (ValueError, TypeError, KeyError):
            continue

    return holdings if holdings else None


# ──────────────────────────────────────────────────────────────────────────────
# STREAMLIT PAGE
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Data Import", page_icon="📤", layout="wide")
st.markdown(PAGE_CSS, unsafe_allow_html=True)

st.markdown("# 📤 Data Import")
st.caption("Upload holdings and model portfolio CSVs to update your dashboard data.")

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
    "Drag & drop your broker holdings CSV",
    type=["csv"],
    key="holdings_upload",
    help="CSV should have columns: Name, Ticker, Currency, Quantity, Market Value, Sector",
)

if holdings_file is not None:
    raw_bytes = holdings_file.read()
    parsed = parse_holdings_csv(raw_bytes)

    if parsed is None:
        st.error(
            "Could not parse holdings CSV. Please ensure it has columns like: "
            "**Name**, **Ticker**, **Quantity**, **Market Value**. "
            "Optional: Currency, Sector."
        )
        # Show raw preview so user can see what we received
        st.caption("Raw CSV preview:")
        try:
            preview_df = pd.read_csv(io.BytesIO(raw_bytes), encoding="utf-8-sig")
            st.dataframe(preview_df.head(5), use_container_width=True)
            st.caption(f"Detected columns: {', '.join(preview_df.columns.tolist())}")
        except Exception:
            pass
    else:
        # Show preview
        preview_df = pd.DataFrame(parsed)
        total_value = preview_df["market_value"].sum()
        num_sectors = preview_df["sector"].nunique()

        st.success(
            f"Parsed **{len(parsed)} positions** · "
            f"**${total_value:,.0f}** total · "
            f"**{num_sectors} sectors**"
        )

        st.dataframe(
            preview_df[["ticker", "name", "currency", "quantity", "market_value", "sector"]],
            use_container_width=True,
            height=300,
            hide_index=True,
        )

        # Confirm import
        if st.button("✅ Confirm Holdings Import", type="primary",
                      key="confirm_holdings"):
            report_date = datetime.now().strftime("%Y-%m-%d")
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
# SECTION 2 — MODEL IMPORT
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-header">Upload Model Portfolio</div>',
            unsafe_allow_html=True)

model_col1, model_col2 = st.columns([1, 2])

with model_col1:
    tab_name = st.text_input(
        "Model tab name",
        placeholder="e.g. NA Div Model",
        help="Short name that appears as a tab on the dashboard",
    )

with model_col2:
    model_file = st.file_uploader(
        "Drag & drop model CSV",
        type=["csv"],
        key="model_upload",
        help="CSV with columns: Name, Ticker (col 3), Target % (col 4)",
    )

if model_file is not None:
    raw_bytes = model_file.read()

    try:
        constituents, cash_pct, model_name = parse_model_csv(raw_bytes)
    except Exception as e:
        st.error(f"Error parsing model CSV: {e}")
        constituents = []

    if not constituents:
        st.error("No constituents found in the CSV. Check the format.")
        # Show raw preview
        st.caption("Raw CSV preview:")
        try:
            preview_df = pd.read_csv(io.BytesIO(raw_bytes), encoding="utf-8-sig")
            st.dataframe(preview_df.head(5), use_container_width=True)
        except Exception:
            pass
    else:
        # Enrich with sector/currency
        constituents = enrich_constituents(constituents)

        total_equity = round(sum(c["target_pct"] for c in constituents), 2)
        preview_df = pd.DataFrame(constituents)

        # Count unmatched sectors
        unmatched = preview_df[preview_df["sector"] == "Other"]

        st.success(
            f"Parsed **{len(constituents)} constituents** · "
            f"**{total_equity}% equity** · **{cash_pct}% cash** · "
            f"Model: {model_name}"
        )

        if not unmatched.empty:
            st.warning(
                f"{len(unmatched)} ticker(s) not found in current holdings — "
                f"sector set to 'Other': {', '.join(unmatched['ticker'].tolist())}"
            )

        st.dataframe(
            preview_df[["ticker", "name", "target_pct", "sector", "currency"]],
            use_container_width=True,
            height=300,
            hide_index=True,
        )

        # Determine output filename
        effective_tab = tab_name.strip() if tab_name and tab_name.strip() else model_name
        if not effective_tab:
            effective_tab = "imported_model"
        slug = re.sub(r"[^a-z0-9]+", "_", effective_tab.lower()).strip("_")
        out_path = MODELS_DIR / f"{slug}.json"

        # Check if overwriting
        if out_path.exists():
            st.info(f"This will **overwrite** `{out_path.name}`")

        if st.button("✅ Confirm Model Import", type="primary",
                      key="confirm_model"):
            if not effective_tab:
                st.error("Please enter a model tab name.")
            else:
                MODELS_DIR.mkdir(exist_ok=True)
                output = {
                    "_meta": {
                        "last_updated": datetime.now().strftime("%Y-%m-%d"),
                        "source": f"Imported from {model_file.name}",
                        "model_name": model_name or effective_tab,
                        "tab_name": effective_tab,
                        "total_equity_pct": total_equity,
                        "cash_pct": cash_pct,
                        "num_constituents": len(constituents),
                    },
                    "constituents": constituents,
                }
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(output, f, indent=2)
                st.cache_data.clear()
                st.success(f"Saved {len(constituents)} constituents to {out_path.name}")
                st.balloons()
