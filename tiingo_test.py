import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import os
import time

load_dotenv()
API_KEY = os.getenv("TIINGO_KEY")

if not API_KEY:
    raise ValueError("TIINGO_KEY not found in environment variables.")

BASE_URL = "https://api.tiingo.com/tiingo/daily"


# -----------------------------
# Load Tickers from CSV
# -----------------------------
def load_tickers(csv_path):
    df = pd.read_csv(csv_path)

    # Get first column regardless of name
    tickers = df.iloc[:, 0].dropna().unique().tolist()

    # Clean tickers for Tiingo (BRK.B -> BRK-B)
    tickers = [str(t).strip().replace(".", "-") for t in tickers]

    return tickers


# -----------------------------
# Fetch Data from Tiingo
# -----------------------------
def fetch_tiingo_data(ticker, start_date="2000-01-01"):
    url = f"{BASE_URL}/{ticker}/prices"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Token {API_KEY}"
    }

    params = {
        "startDate": start_date,
        "format": "json",
        "resampleFreq": "daily"
    }

    response = requests.get(url, headers=headers, params=params)

    # Raise error if request failed
    response.raise_for_status()

    data = response.json()

    if not data:
        print(f"No data returned for {ticker}")
        return None

    df = pd.DataFrame(data)

    # Optional: Add daily returns
    if "adjClose" in df.columns:
        df["returns"] = df["adjClose"].pct_change()

    return df


# -----------------------------
# Save Data
# -----------------------------
def save_data(df, ticker):
    path = Path(f"data/raw/prices/{ticker}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)
    print(f"Saved {ticker} -> {path}")


# -----------------------------
# Main Execution
# -----------------------------
def main():
    tickers = load_tickers("sp500.csv")

    print(f"Loaded {len(tickers)} tickers.")
    print("First 5 tickers:", tickers[:5])

    for i, ticker in enumerate(tickers, 1):
        try:
            print(f"[{i}/{len(tickers)}] Fetching {ticker}...")
            df = fetch_tiingo_data(ticker)

            if df is not None:
                save_data(df, ticker)

            # Prevent rate limit issues
            time.sleep(1)

        except Exception as e:
            print(f"Failed for {ticker}: {e}")
            continue


if __name__ == "__main__":
    main()