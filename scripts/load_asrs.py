"""
NASA ASRS (Aviation Safety Reporting System) data loader.

Downloads and normalises NASA ASRS incident narratives into the same schema
used by the synthetic document generator so both data sources can be ingested
by the same pipeline.

--- How to get the data ---

1. Go to https://asrs.arc.nasa.gov/search/database.html
2. Click "Database Online" → "ASRS Database Search"
3. Leave filters blank to pull all report types, or filter to:
   - Report Type: Incident
   - Aircraft Category: Air Transport (for commercial aerospace)
4. Click "Submit Query"
5. On the results page click "Download All Records (CSV)"
6. Save the file to:  data/raw/asrs_data.csv

The CSV contains columns including:
  ACN, Date, Local Time Of Day, Callback, Report Type,
  Narrative, Synopsis, Assessments, ...

This script reads that CSV and writes data/raw/asrs.jsonl in the same
{doc_id, doc_type, text, date, ...} schema used by synthetic docs.

Usage:
    python scripts/load_asrs.py
    python scripts/load_asrs.py --input data/raw/asrs_data.csv --limit 500
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import load_config

# Columns we care about — ASRS CSV has many more but these cover text + metadata
_KEEP_COLS = {
    "ACN": "acn",
    "Date": "date_raw",
    "Report Type": "report_type",
    "Narrative": "narrative",
    "Synopsis": "synopsis",
}


def normalise_date(raw: str) -> str:
    """Convert ASRS date strings (e.g. '200301', 'January 2003') to YYYY-MM-DD."""
    raw = str(raw).strip()
    # YYYYMM format — common in older ASRS exports
    m = re.fullmatch(r"(\d{4})(\d{2})", raw)
    if m:
        return f"{m.group(1)}-{m.group(2)}-01"
    # Try pandas as a fallback
    try:
        return pd.to_datetime(raw).strftime("%Y-%m-%d")
    except Exception:
        return raw


def build_text(row: pd.Series) -> str:
    """Combine narrative and synopsis into a single text field."""
    parts = []
    for col in ("narrative", "synopsis"):
        val = str(row.get(col, "")).strip()
        if val and val.lower() not in ("", "nan", "none"):
            parts.append(val)
    return "\n\n".join(parts)


def load_asrs(input_path: Path, limit: int | None, out_path: Path) -> None:
    print(f"Reading {input_path} ...")
    df = pd.read_csv(input_path, encoding="latin-1", low_memory=False)

    # Rename columns we care about; drop everything else
    rename = {k: v for k, v in _KEEP_COLS.items() if k in df.columns}
    df = df.rename(columns=rename)[list(rename.values())]

    if limit:
        df = df.head(limit)

    print(f"  {len(df)} rows after limit filter")

    out_file = out_path / "asrs.jsonl"
    written = 0
    with open(out_file, "w") as f:
        for _, row in df.iterrows():
            text = build_text(row)
            if len(text) < 50:
                continue  # skip nearly-empty rows

            doc = {
                "doc_id": f"ASRS-{str(row.get('acn', '')).strip()}",
                "doc_type": "incident_report",
                "source": "NASA_ASRS",
                "report_type": str(row.get("report_type", "")).strip(),
                "date": normalise_date(row.get("date_raw", "")),
                "text": text,
            }
            f.write(json.dumps(doc) + "\n")
            written += 1

    print(f"  Wrote {written} documents → {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalise NASA ASRS CSV into JSONL")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to ASRS CSV (default: data/raw/asrs_data.csv)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N rows (useful for testing)",
    )
    args = parser.parse_args()

    cfg = load_config()
    raw_path = Path(cfg["paths"]["raw_data"])
    raw_path.mkdir(parents=True, exist_ok=True)

    input_path = args.input or raw_path / "asrs_data.csv"
    if not input_path.exists():
        print(f"ERROR: {input_path} not found.")
        print(__doc__)
        sys.exit(1)

    load_asrs(input_path, args.limit, raw_path)
    print("Done.")


if __name__ == "__main__":
    main()
