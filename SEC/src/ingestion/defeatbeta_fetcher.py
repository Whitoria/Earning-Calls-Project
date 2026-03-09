import pandas as pd
import os
import time
from defeatbeta_api.data.ticker import Ticker

# Top 10 S&P 500 Tickers
TICKERS = [
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", 
    "AVGO", "META", "TSLA", "BRK-B", "LLY"
]

DATA_DIR = "SEC/data"
OUTPUT_FILE = f"{DATA_DIR}/defeatbeta_financials.parquet"

def fetch_financials():
    all_data = []
    
    for symbol in TICKERS:
        print(f"Fetching DefeatBeta financials for {symbol}...")
        try:
            # Note: BRK-B is often BRK.B in some APIs. DefeatBeta usually handles standard tickers.
            # If it fails, we fall back or skip.
            tkr = Ticker(symbol)
            
            # Fetch quarterly income statements using the correct method name and call it
            stmt = tkr.quarterly_income_statement()
            
            if stmt is not None and hasattr(stmt, 'df'):
                inc_stmt = stmt.df()
            else:
                inc_stmt = None
            
            if inc_stmt is not None and not inc_stmt.empty:
                # The dataframe has metrics in the 'Breakdown' column and dates as other columns
                metric_col = 'Breakdown' if 'Breakdown' in inc_stmt.columns else inc_stmt.columns[0]
                
                # We don't need 'TTM' (Trailing Twelve Months) for quarterly discrete correlation
                cols_to_keep = [c for c in inc_stmt.columns if c != 'TTM']
                inc_stmt = inc_stmt[cols_to_keep]
                
                # Add ticker column
                inc_stmt['ticker'] = symbol
                
                # Melt dates into rows
                melted = inc_stmt.melt(id_vars=['ticker', metric_col], var_name='reported_date', value_name='value')
                
                # Standardize metric column name
                melted = melted.rename(columns={metric_col: 'metric'})
                
                all_data.append(melted)
            
            time.sleep(1) # Be polite to the API
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")

    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        # Clean up dates and values for Parquet compatibility
        df['reported_date'] = pd.to_datetime(df['reported_date'], errors='coerce')
        df = df.dropna(subset=['reported_date'])
        
        # DefeatBeta returns decimal.Decimal which PyArrow hates. Coerce to float.
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Save structural data
        df.to_parquet(OUTPUT_FILE, index=False)
        print(f"DefeatBeta financials saved to {OUTPUT_FILE} ({len(df)} records)")
    else:
        print("No DefeatBeta data retrieved.")

if __name__ == "__main__":
    fetch_financials()
