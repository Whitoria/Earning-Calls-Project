import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("TIINGO_KEY")

tickers = ["AAPL", "MSFT", "AMZN"]

BASE_URL = "https://api.tiingo.com/tiingo/daily"

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
    response.raise_for_status()

    data = response.json()
    df = pd.DataFrame(data)

    return df


def save_data(df, ticker):
    path = Path(f"data/raw/prices/{ticker}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False)
    print(f"Saved {ticker} to {path}")


for ticker in tickers:
    df = fetch_tiingo_data(ticker)
    save_data(df, ticker)
