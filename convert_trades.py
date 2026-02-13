"""
================================================================================
 BLOCK TRADE EXCEL → JSON CONVERTER
 ────────────────────────────────────
 Converts the block trade Excel export into block_trades.json for the dashboard.

 USAGE:
   python convert_trades.py path/to/Book2.xlsx
   python convert_trades.py path/to/Book2.xlsx --output custom_output.json

 The script:
   1. Reads the Excel file (skips the header row)
   2. Normalizes "Bought"/"Sold" → "buy"/"sell"
   3. Maps tickers to yfinance format (appends .TO for CAD, etc.)
   4. Detects switch trades from "Part of Switch" in notes
   5. Auto-generates trade IDs and switch IDs
   6. Outputs block_trades.json
================================================================================
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


# ── Ticker mapping to yfinance format ──

def map_ticker_to_yfinance(symbol: str, currency: str) -> str:
    """Convert raw Excel ticker to yfinance-compatible format."""
    if not symbol or pd.isna(symbol):
        return str(symbol) if symbol else ""

    symbol = str(symbol).strip()

    # Already has .TO suffix
    if symbol.endswith(".TO"):
        return symbol

    # Known US-listed tickers (no suffix needed)
    us_tickers = {"NVO", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "ABBV",
                  "KO", "PG", "GLD", "WCN", "IBIT", "ETHA"}
    if symbol.upper() in us_tickers or currency == "USD":
        return symbol.upper()

    # Fund codes (FID670, FID2210, etc.) — not tradeable on exchanges
    if re.match(r'^FID\d+$', symbol, re.IGNORECASE):
        return symbol.upper()

    # Handle .B suffix → -B.TO (e.g., IDIV.B → IDIV-B.TO)
    if ".B" in symbol.upper():
        base = symbol.upper().replace(".B", "-B")
        return f"{base}.TO"

    # Handle .UN suffix → -UN.TO (trust units)
    if ".UN" in symbol.upper():
        base = symbol.upper().replace(".UN", "-UN")
        return f"{base}.TO"

    # Default: CAD ticker gets .TO suffix
    if currency == "CAD":
        return f"{symbol.upper()}.TO"

    return symbol.upper()


def detect_switches(trades: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Detect switch trades from notes containing "Part of Switch".
    Groups them by the switch description and assigns switch IDs.
    Returns updated trades list and switches list.
    """
    switch_pattern = re.compile(r'part of switch', re.IGNORECASE)
    switch_groups = {}  # rationale → list of trade IDs
    switch_id_counter = 1

    for trade in trades:
        notes = trade.get("notes", "") or ""
        if switch_pattern.search(notes):
            # Use the notes text (minus "Part of Switch") as the grouping key
            rationale = re.sub(r'Part of Switch\s*[-—]?\s*', '', notes,
                               flags=re.IGNORECASE).strip()
            if not rationale:
                rationale = notes

            if rationale not in switch_groups:
                sid = f"SW-{datetime.now().year}-{switch_id_counter:03d}"
                switch_groups[rationale] = {
                    "switch_id": sid,
                    "label": _make_switch_label(rationale),
                    "rationale": rationale,
                    "sell_trade_ids": [],
                    "buy_trade_ids": [],
                    "status": "completed",
                }
                switch_id_counter += 1

            sw = switch_groups[rationale]
            trade["switch_id"] = sw["switch_id"]
            if trade["type"] == "buy":
                sw["buy_trade_ids"].append(trade["id"])
            else:
                sw["sell_trade_ids"].append(trade["id"])

    # Also link continuation rows (no date, immediately after a switch trade)
    for i, trade in enumerate(trades):
        if trade.get("switch_id"):
            continue
        # Check if this is a continuation row (no date/shares) near a switch
        if trade.get("date") is None and trade.get("shares") is None:
            # Look backward for the nearest switch trade
            for j in range(i - 1, max(i - 3, -1), -1):
                if trades[j].get("switch_id"):
                    sw_id = trades[j]["switch_id"]
                    trade["switch_id"] = sw_id
                    # Find the switch group and add this trade
                    for sw in switch_groups.values():
                        if sw["switch_id"] == sw_id:
                            if trade["type"] == "sell":
                                sw["sell_trade_ids"].append(trade["id"])
                            else:
                                sw["buy_trade_ids"].append(trade["id"])
                            break
                    break

    return trades, list(switch_groups.values())


def _make_switch_label(rationale: str) -> str:
    """Generate a short label from the switch rationale."""
    # Try to extract the key entity
    rationale_lower = rationale.lower()
    if "fidelity" in rationale_lower:
        return "Fidelity International Exit"
    # Default: first 40 chars
    return rationale[:40].strip()


def convert_excel_to_json(excel_path: str, output_path: str = None):
    """Main conversion function."""
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    if output_path is None:
        output_path = Path(excel_path).parent / "block_trades.json"
    else:
        output_path = Path(output_path)

    # Read Excel — the first row is actually column headers
    df = pd.read_excel(excel_path)

    # The actual headers are in row 0 of the data
    # Columns: Date, Transaction, Amount/Par, Description, Symbol, Price,
    #          Currency, Yield, Approximate Market Value, Notes
    col_map = {
        df.columns[0]: "date",
        df.columns[1]: "transaction",
        df.columns[2]: "shares",
        df.columns[3]: "name",
        df.columns[4]: "symbol",
        df.columns[5]: "price",
        df.columns[6]: "currency",
        df.columns[7]: "yield_pct",
        df.columns[8]: "market_value",
        df.columns[9]: "notes",
    }
    df = df.rename(columns=col_map)

    # Skip the header-description row (row 0) and any blank rows
    df = df.iloc[1:]  # skip the "Date, Transaction, ..." label row
    df = df[~(df["date"].isna() & df["transaction"].isna() &
              df["shares"].isna() & df["name"].isna() &
              df["symbol"].isna())].copy()

    # Also skip fully empty rows
    df = df.dropna(how="all")

    if df.empty:
        print("No trades found in the Excel file.")
        return

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")

    # Determine period
    valid_dates = df["date"].dropna()
    period_start = valid_dates.min().strftime("%Y-%m-%d") if not valid_dates.empty else None
    period_end = valid_dates.max().strftime("%Y-%m-%d") if not valid_dates.empty else None

    trades = []
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        trade_type = None
        txn = str(row.get("transaction", "")).strip().lower()
        if txn in ("bought", "buy"):
            trade_type = "buy"
        elif txn in ("sold", "sell"):
            trade_type = "sell"
        elif pd.isna(row.get("transaction")):
            # Continuation row — infer from context
            trade_type = "sell"  # continuation rows are typically the sell side
        else:
            continue

        currency = str(row.get("currency", "CAD")).strip() if pd.notna(row.get("currency")) else "CAD"
        symbol_raw = str(row.get("symbol", "")).strip() if pd.notna(row.get("symbol")) else ""
        symbol = map_ticker_to_yfinance(symbol_raw, currency)

        date_val = row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else None
        shares = int(row["shares"]) if pd.notna(row.get("shares")) else None
        price = float(row["price"]) if pd.notna(row.get("price")) else None
        mv = round(float(row["market_value"])) if pd.notna(row.get("market_value")) else None
        yield_pct = float(row["yield_pct"]) if pd.notna(row.get("yield_pct")) else None
        notes = str(row.get("notes", "")).strip() if pd.notna(row.get("notes")) else None
        name = str(row.get("name", symbol)).strip() if pd.notna(row.get("name")) else symbol

        trades.append({
            "id": f"BT-{datetime.now().year}-{idx:04d}",
            "date": date_val,
            "type": trade_type,
            "symbol": symbol,
            "name": name,
            "shares": shares,
            "price": price,
            "currency": currency,
            "market_value": mv,
            "yield_pct": yield_pct,
            "notes": notes,
            "switch_id": None,
        })

    # Detect and link switch trades
    trades, switches = detect_switches(trades)

    # Build output
    output = {
        "_meta": {
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "source": f"Converted from {excel_path.name}",
            "period_start": period_start,
            "period_end": period_end,
            "notes": "Block trade activity",
        },
        "trades": trades,
        "switches": switches,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Converted {len(trades)} trades ({len(switches)} switches) -> {output_path}")
    for t in trades:
        status = f"[{t['type'].upper():4s}]"
        date = t["date"] or "     —    "
        mv = f"${t['market_value']:>12,.0f}" if t["market_value"] else "           —"
        sw = f"  [SW: {t['switch_id']}]" if t["switch_id"] else ""
        print(f"  {status} {date}  {t['symbol']:<12s} {mv}{sw}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert block trade Excel export to JSON for the dashboard"
    )
    parser.add_argument("excel_file", help="Path to the block trades Excel file")
    parser.add_argument("--output", "-o", default=None,
                        help="Output JSON path (default: block_trades.json in same dir)")
    args = parser.parse_args()
    convert_excel_to_json(args.excel_file, args.output)
