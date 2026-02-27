"""
CSV -> JSON converter for the Model Portfolio.

Usage:
    python convert_model.py "path/to/Model Constituents.csv"

Produces model_portfolio.json in the same directory as this script.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

SUFFIX_MAP = {
    "-T": ".TO",   # TSX
    "-N": "",      # NYSE
    "-O": "",      # NASDAQ
}


def convert_ticker(raw: str) -> str:
    """Convert 'ARE-T' -> 'ARE.TO', 'GOOGL-O' -> 'GOOGL', 'XOM-N' -> 'XOM'."""
    raw = raw.strip()
    for suffix, replacement in SUFFIX_MAP.items():
        if raw.endswith(suffix):
            base = raw[: -len(suffix)]
            return base + replacement
    return raw


def convert_csv_to_json(csv_path: str, output_path: str | None = None):
    csv_path = Path(csv_path)
    if output_path is None:
        output_path = Path(__file__).parent / "model_portfolio.json"
    else:
        output_path = Path(output_path)

    constituents = []
    cash_pct = 0.0
    model_name = ""

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue

            # Model name from the second row (first data row, not header)
            if row[0].strip() and row[0].strip() != "Name" and not model_name:
                model_name = row[0].strip()

            # Cash row: column 1 == "Currency/Cash"
            if len(row) >= 5 and row[1].strip() == "Currency/Cash":
                try:
                    cash_pct = float(row[4].strip())
                except (ValueError, IndexError):
                    pass
                continue

            # Equity rows: column 3 has a ticker like "ARE-T"
            if len(row) < 5 or not row[3].strip():
                continue
            raw_ticker = row[3].strip()
            if raw_ticker == "Ticker":
                continue  # header row

            name = row[2].strip()
            try:
                tgt_pct = float(row[4].strip())
            except (ValueError, IndexError):
                continue
            if tgt_pct == 0:
                continue  # skip zero-weight entries

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

    output = {
        "_meta": {
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "source": f"Converted from {csv_path.name}",
            "model_name": model_name or "Model Portfolio",
            "total_equity_pct": round(sum(c["target_pct"] for c in constituents), 2),
            "cash_pct": cash_pct,
            "num_constituents": len(constituents),
        },
        "constituents": constituents,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Converted {len(constituents)} constituents -> {output_path}")
    print(f"  Model: {output['_meta']['model_name']}")
    print(f"  Cash: {cash_pct}%, Equity: {output['_meta']['total_equity_pct']}%")
    for c in constituents:
        print(f"  {c['ticker']:<15} {c['target_pct']:>5}%  ({c['name'][:40]})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert model CSV to JSON")
    parser.add_argument("csv_file", help="Path to the Model Constituents CSV")
    parser.add_argument("-o", "--output", help="Output JSON path (default: model_portfolio.json)")
    args = parser.parse_args()
    convert_csv_to_json(args.csv_file, args.output)
