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


def parse_holdings_file(file_bytes: bytes, filename: str) -> list[dict] | None:
    """Parse a holdings file (CSV or Excel) from broker export.

    Expected columns (flexible matching):
      - Name / Security Name
      - Ticker / Symbol
      - Currency / Curr
      - Quantity / Shares / Units
      - Market Value / Value
      - Sector / Asset Class

    Returns list of holdings dicts or None if parsing fails.
    """
    # Read into DataFrame based on file type
    try:
        if filename.lower().endswith(".xlsx") or filename.lower().endswith(".xls"):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            text = file_bytes.decode("utf-8-sig")
            df = pd.read_csv(io.StringIO(text))
    except Exception:
        return None

    # Normalize column names for flexible matching
    col_map = {}
    for col in df.columns:
        cl = str(col).strip().lower()
        if cl in ("name", "security name", "security", "description"):
            col_map["name"] = col
        elif cl in ("ticker", "symbol", "ticker symbol"):
            col_map["ticker"] = col
        elif cl in ("currency", "curr", "ccy"):
            col_map["currency"] = col
        elif cl in ("quantity", "shares", "units", "qty"):
            col_map["quantity"] = col
        elif cl in ("market value", "value", "mkt value", "market_value",
                     "mktval", "market val"):
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
            quantity = float(
                str(row[col_map["quantity"]]).replace(",", ""))
            market_value = float(
                str(row[col_map["market_value"]])
                .replace(",", "").replace("$", ""))

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
    parsed = parse_holdings_file(raw_bytes, holdings_file.name)

    if parsed is None:
        st.error(
            "Could not parse holdings file. Please ensure it has columns like: "
            "**Name**, **Ticker**, **Quantity**, **Market Value**. "
            "Optional: Currency, Sector."
        )
        # Show raw preview so user can see what we received
        st.caption("Raw file preview:")
        try:
            if holdings_file.name.lower().endswith((".xlsx", ".xls")):
                preview_df = pd.read_excel(io.BytesIO(raw_bytes))
            else:
                preview_df = pd.read_csv(
                    io.BytesIO(raw_bytes), encoding="utf-8-sig")
            st.dataframe(preview_df.head(5), use_container_width=True)
            st.caption(
                f"Detected columns: {', '.join(str(c) for c in preview_df.columns.tolist())}")
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
            preview_df[["ticker", "name", "currency", "quantity",
                         "market_value", "sector"]],
            use_container_width=True,
            height=300,
            hide_index=True,
        )

        # Confirm import
        if st.button("Confirm Holdings Import", type="primary",
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
