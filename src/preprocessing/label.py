import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from transformers import pipeline
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

price_cache: dict[str, pd.DataFrame] = {}

def load_tiingo_prices(ticker: str, prices_dir: str) -> pd.DataFrame | None:
    if ticker in price_cache:
        return price_cache[ticker]
    
    for fname in [f"{ticker}.csv", f"{ticker.lower()}.csv"]:
        path = Path(prices_dir) / fname
        if path.exists():
            break
        else:
            logger.warning(f"No Tiingo price file found for {ticker} in {prices_dir}")
            return None
        
    df = pd.read_csv(path)
        
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
    df = df.sort_values("date").set_index("date")

    if "adjClose" not in df.columns:
        logger.warning(f"{ticker}: no adjClose column found, falling back to 'close'")
        df["adjClose"] = df["close"]

    price_cache[ticker] = df
    logger.debug(f"Loaded {len(df)} rows for {ticker} from {path}")
    return df

def fetch_post_earnings_return(
        ticker: str,
        earnings_date: str,
        days_after: int,
        prices_dir: str
) -> float | None:
    
    prices = load_tiingo_prices(ticker, prices_dir)
    if prices is None:
        return None
    
    dt = None
    for fmt in ["%B %d, %Y", "%Y-%m-%d", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S.%fZ"]:
        try:
            dt = pd.Timestamp(datetime.strptime(earnings_date.strip(), fmt)).normalize()
            break
        except ValueError:
            continue

    if dt is None:
        logger.warning(f"Could not parse earnings date: '{earnings_date}")
        return None
        
    future = prices[prices.index >= dt]["adjClose"]

    if len(future) < days_after + 1:
        logger.warning(
            f"{ticker} @ {earnings_date}: only {len(future)} trading days available "
            f"after earnings (need {days_after + 1})"
        )
        return None
    
    price_before = future.iloc[0]
    price_after = future.iloc[days_after]

    if price_before == 0:
        return None
    
    return float((price_after-price_before) / price_before)

def label_by_returns(df: pd.DataFrame, days_after: int = 1, prices_dir: str = "data/prices") -> pd.DataFrame:
    pairs = df[["ticker", "date"]].drop_duplicates()
    returns = {}

    for _, row in pairs.iterrows():
        ticker, date = row["ticker"], row["date"]
        logger.info(f"Fetching return for {ticker} on {date}...")
        r = fetch_post_earnings_return(ticker, date, days_after, prices_dir)
        returns[(ticker, date)] = r
        logger.info(f"  → {r:.4f}" if r is not None else "  → N/A")
    
    df["stock_return"] = df.apply(lambda r: returns.get((r["ticker"], r["date"])), axis=1)
    df = df.dropna(subset=["stock_return"])
    
    if df.empty:
        logger.error("No returns could be fetched. Check tickers and dates.")
        return df
    
    # Assign labels using tertile thresholds
    low  = df["stock_return"].quantile(0.33)
    high = df["stock_return"].quantile(0.67)
    
    def assign_label(r):
        if r < low:  return "negative"
        if r > high: return "positive"
        return "neutral"
    
    df["label"] = df["stock_return"].apply(assign_label)
    logger.info(f"Return thresholds: negative < {low:.3f} < neutral < {high:.3f} < positive")
    logger.info(f"Label distribution:\n{df['label'].value_counts()}")
    return df
