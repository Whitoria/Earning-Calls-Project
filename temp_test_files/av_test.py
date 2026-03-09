import requests
import time
import pandas as pd
from datetime import datetime
from io import StringIO
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ALPHA_VANTAGE_KEY1")
tickers = ["IBM", "AAPL", "AMZN"]

# results = []

# today = datetime.now().strftime("%Y-%m-%d")

# for ticker in tickers:
#     url = (
#         f"https://www.alphavantage.co/query?"
#         f"function=TIME_SERIES_DAILY"
#         f"&symbol={ticker}"
#         f"&apikey={api_key}"
#         f"&outputsize=full"
#         f"&datatype=csv"
#     )

#     r = requests.get(url)

#     df = pd.read_csv(StringIO(r.text))

#     file_path = Path(f"data/raw/prices/{ticker}.csv")
#     file_path.parent.mkdir(parents=True, exist_ok=True)

#     df.to_csv(file_path, index=False)

#     print("Saved to:", file_path.resolve())

#     time.sleep(12)

url = 'https://www.alphavantage.co/query?function=EARNINGS_CALL_TRANSCRIPT&symbol=IBM&quarter=2024Q1&apikey=demo'

url = (
    f"https://www.alphavantage.co/query?"
    f"function=EARNINGS_CALL_TRANSCRIPT"
    f"&symbol={tickers[1]}"
    f"&apikey={api_key}"
    f"&quarter=2024Q1"
)

r = requests.get(url)
data = r.json()

print(data)
