"""
This script:
  1. Walks the MAEC folder, parses ticker + date from each folder name
  2. Splits each transcript into sentences
  3. Fetches the post-earnings adjClose return from your Tiingo CSVs
  4. Assigns positive / neutral / negative labels by return tertile
  5. Outputs a labeled CSV ready for train.py

Usage:
    python src/ingest_maec.py \
        --maec_dir  "data/MAEC-A-Multimodal-.../MAEC_Dataset" \
        --prices_dir data/prices \
        --output     data/labeled.csv \
        --days_after 1

Example with AAPL Tiingo file at data/prices/AAPL.csv:
    The folder 20150128_AAPL will be parsed as ticker=AAPL, date=2015-01-28.
    The script will look up AAPL's adjClose on 2015-01-28 and the next trading day,
    compute the 1-day return, and label every sentence from that call accordingly.
"""

import re
import argparse
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# TIINGO PRICE LOADER  (shared with label_data.py)
# ─────────────────────────────────────────────

_price_cache: dict[str, pd.DataFrame] = {}


def _load_tiingo(ticker: str, prices_dir: str) -> pd.DataFrame | None:
    """Load and cache a Tiingo CSV. Returns a DataFrame indexed by date."""
    if ticker in _price_cache:
        return _price_cache[ticker]

    for fname in [f"{ticker}.csv", f"{ticker.lower()}.csv"]:
        path = Path(prices_dir) / fname
        if path.exists():
            df = pd.read_csv(path)
            df["date"] = (
                pd.to_datetime(df["date"], utc=True)
                .dt.tz_localize(None)
                .dt.normalize()
            )
            df = df.sort_values("date").set_index("date")
            if "adjClose" not in df.columns:
                logger.warning(f"{ticker}: no adjClose column, falling back to 'close'")
                df["adjClose"] = df["close"]
            _price_cache[ticker] = df
            return df

    logger.warning(f"No Tiingo CSV found for {ticker} in {prices_dir}/")
    return None


def _post_earnings_return(ticker: str, date: pd.Timestamp, days_after: int, prices_dir: str) -> float | None:
    """
    Return the adjClose-based N-trading-day return starting from `date`.
    iloc[0] = close on (or first trading day after) earnings date
    iloc[N] = close N trading days later
    """
    prices = _load_tiingo(ticker, prices_dir)
    if prices is None:
        return None

    future = prices.loc[prices.index >= date, "adjClose"]

    if len(future) < days_after + 1:
        logger.warning(
            f"{ticker} @ {date.date()}: only {len(future)} rows after earnings "
            f"(need {days_after + 1}) — skipping"
        )
        return None

    p0 = future.iloc[0]
    p1 = future.iloc[days_after]
    return float((p1 - p0) / p0) if p0 != 0 else None


# ─────────────────────────────────────────────
# TEXT SPLITTING
# ─────────────────────────────────────────────

# Sentence-ending punctuation followed by whitespace (or end of string)
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')


def split_sentences(text: str, min_chars: int = 30) -> list[str]:
    """
    Split raw transcript text into individual sentences.
    Filters out very short fragments (boilerplate headers, page numbers, etc.)

    min_chars: discard any sentence shorter than this — tunable.
    """
    # Normalise whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = _SENT_SPLIT.split(text)
    return [s.strip() for s in sentences if len(s.strip()) >= min_chars]


# ─────────────────────────────────────────────
# FOLDER PARSER
# ─────────────────────────────────────────────

def parse_folder_name(folder_name: str) -> tuple[str, pd.Timestamp] | None:
    """
    Parse a MAEC folder name like '20150225_LMAT' into (ticker, date).
    Returns None if the folder name doesn't match the expected pattern.
    """
    match = re.fullmatch(r'(\d{8})_([A-Z0-9\.\-]+)', folder_name)
    if not match:
        return None
    date_str, ticker = match.group(1), match.group(2)
    try:
        dt = pd.Timestamp(datetime.strptime(date_str, "%Y%m%d")).normalize()
    except ValueError:
        return None
    return ticker, dt


# ─────────────────────────────────────────────
# MAIN INGESTION PIPELINE
# ─────────────────────────────────────────────

def ingest(maec_dir: str, prices_dir: str, output: str, days_after: int, min_chars: int):
    maec_path = Path(maec_dir)
    if not maec_path.exists():
        raise FileNotFoundError(f"MAEC directory not found: {maec_dir}")

    call_folders = sorted([f for f in maec_path.iterdir() if f.is_dir()])
    logger.info(f"Found {len(call_folders)} call folders in {maec_dir}")

    # ── Pass 1: collect raw data + returns ──────────────────────────────────
    rows = []
    return_map: dict[tuple, float | None] = {}   # (ticker, date) → return
    skipped_parse   = 0
    skipped_no_text = 0

    for folder in call_folders:
        parsed = parse_folder_name(folder.name)
        if parsed is None:
            logger.warning(f"Skipping unrecognised folder name: {folder.name}")
            skipped_parse += 1
            continue

        ticker, date = parsed
        text_file = folder / "text.txt"

        if not text_file.exists():
            logger.warning(f"No text.txt in {folder.name} — skipping")
            skipped_no_text += 1
            continue

        raw_text = text_file.read_text(encoding="utf-8", errors="replace")
        sentences = split_sentences(raw_text, min_chars=min_chars)

        if not sentences:
            logger.warning(f"{folder.name}: no sentences extracted after filtering")
            continue

        # Fetch return once per call (not once per sentence)
        key = (ticker, date)
        if key not in return_map:
            r = _post_earnings_return(ticker, date, days_after, prices_dir)
            return_map[key] = r
            status = f"{r:+.4f}" if r is not None else "N/A (no price data)"
            logger.info(f"  {folder.name:25s}  {len(sentences):4d} sentences   return={status}")

        ret = return_map[key]

        for i, sentence in enumerate(sentences):
            rows.append({
                "folder":       folder.name,
                "ticker":       ticker,
                "date":         date.strftime("%Y-%m-%d"),
                "sentence_idx": i,
                "text":         sentence,
                "stock_return": ret,
            })

    if not rows:
        logger.error("No data extracted. Check your MAEC directory path.")
        return

    df = pd.DataFrame(rows)
    total_calls    = df["folder"].nunique()
    priced_calls   = df.dropna(subset=["stock_return"])["folder"].nunique()
    logger.info(
        f"\n{'─'*50}\n"
        f"  Total call folders processed : {total_calls}\n"
        f"  Calls with Tiingo price data : {priced_calls}\n"
        f"  Calls missing price data     : {total_calls - priced_calls}\n"
        f"  Skipped (bad folder name)    : {skipped_parse}\n"
        f"  Skipped (no text.txt)        : {skipped_no_text}\n"
        f"  Total sentences              : {len(df)}\n"
        f"{'─'*50}"
    )

    # ── Pass 2: assign labels from return tertiles ──────────────────────────
    df_priced = df.dropna(subset=["stock_return"]).copy()

    if df_priced.empty:
        logger.error(
            "No calls could be matched to Tiingo price data.\n"
            "  → Make sure your Tiingo CSVs are named <TICKER>.csv and placed in "
            f"'{prices_dir}/'\n"
            "  → Example: if MAEC has folder '20150225_LMAT', you need 'data/prices/LMAT.csv'"
        )
        # Save unlabeled output anyway so the user can inspect it
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output.replace(".csv", "_UNLABELED.csv"), index=False)
        logger.info(f"Saved unlabeled data to {output.replace('.csv', '_UNLABELED.csv')}")
        return

    # Tertile thresholds computed across the whole dataset
    low  = df_priced["stock_return"].quantile(0.33)
    high = df_priced["stock_return"].quantile(0.67)

    def _label(r: float) -> str:
        if r < low:  return "negative"
        if r > high: return "positive"
        return "neutral"

    df_priced["label"] = df_priced["stock_return"].apply(_label)

    logger.info(
        f"Return label thresholds:  negative < {low:+.4f} ≤ neutral ≤ {high:+.4f} < positive"
    )
    logger.info(f"Label distribution:\n{df_priced['label'].value_counts().to_string()}")

    # ── Save ────────────────────────────────────────────────────────────────
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df_priced.to_csv(output, index=False)
    logger.info(f"\nSaved {len(df_priced)} labeled sentences → {output}")

    # Also save a call-level summary (one row per earnings call)
    summary_path = output.replace(".csv", "_call_summary.csv")
    summary = (
        df_priced.groupby(["ticker", "date", "folder"])
        .agg(
            sentence_count=("text", "count"),
            stock_return=("stock_return", "first"),
            label=("label", "first"),
        )
        .reset_index()
        .sort_values(["ticker", "date"])
    )
    summary.to_csv(summary_path, index=False)
    logger.info(f"Saved call-level summary ({len(summary)} calls) → {summary_path}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest MAEC dataset and label sentences using Tiingo stock returns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
# Basic usage — 1-day return window, Tiingo files in data/prices/
python src/ingest_maec.py \\
    --maec_dir  "data/MAEC-A-Multimodal-Aligned-Earnings-Conference-Call-Dataset-for-Financial-Risk-Prediction-master/MAEC_Dataset" \\
    --prices_dir data/prices \\
    --output     data/labeled.csv

# 5-day return window
python src/ingest_maec.py \\
    --maec_dir  "data/MAEC_Dataset" \\
    --prices_dir data/prices \\
    --output     data/labeled_5d.csv \\
    --days_after 5

# AAPL example — verifying a single ticker works before running the full dataset
#   1. Make sure data/prices/AAPL.csv exists (downloaded from Tiingo)
#   2. Make sure MAEC_Dataset contains at least one YYYYMMDD_AAPL folder
#   3. Run with --maec_dir pointing at a test subfolder if you want a quick sanity check
        """,
    )
    parser.add_argument(
        "--maec_dir",
        required=True,
        help=(
            "Path to the MAEC_Dataset folder containing YYYYMMDD_TICKER subfolders.\n"
            r"Example (Windows): data\MAEC-...\MAEC_Dataset"
        ),
    )
    parser.add_argument(
        "--prices_dir",
        default="data/prices",
        help="Folder containing Tiingo CSVs named <TICKER>.csv  (default: data/prices)",
    )
    parser.add_argument(
        "--output",
        default="data/labeled.csv",
        help="Output CSV path for labeled sentences  (default: data/labeled.csv)",
    )
    parser.add_argument(
        "--days_after",
        type=int,
        default=1,
        help=(
            "Number of trading days after the call to measure the stock return.\n"
            "1 = next-day reaction (default). Try 5 for a week-level signal."
        ),
    )
    parser.add_argument(
        "--min_chars",
        type=int,
        default=30,
        help="Minimum character length to keep a sentence (default: 30). "
             "Increase to filter out more boilerplate.",
    )
    args = parser.parse_args()

    ingest(
        maec_dir=args.maec_dir,
        prices_dir=args.prices_dir,
        output=args.output,
        days_after=args.days_after,
        min_chars=args.min_chars,
    )