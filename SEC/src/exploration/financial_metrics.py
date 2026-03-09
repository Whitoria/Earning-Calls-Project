import pandas as pd
import os

DATA_DIR = "SEC/data"
INPUT_FILE = f"{DATA_DIR}/defeatbeta_financials.parquet"
OUTPUT_FILE = f"{DATA_DIR}/financial_growth_metrics.parquet"

# Key metrics we care about for future correlation
TARGET_METRICS = [
    'Net Income',
    'Total Revenue',
    'Operating Income',
    'Gross Profit'
]

def process_metrics():
    if not os.path.exists(INPUT_FILE):
        print(f"File not found: {INPUT_FILE}")
        return

    df = pd.read_parquet(INPUT_FILE)
    
    # Filter for primary target metrics to reduce noise
    df = df[df['metric'].isin(TARGET_METRICS)].copy()
    
    # Sort chronologically
    df = df.sort_values(by=['ticker', 'metric', 'reported_date'])
    
    print(f"Processing {len(df)} core financial records...")

    # Calculate Quarter-over-Quarter (QoQ) and Year-over-Year (YoY) growth
    # YoY is usually a 4-period shift for quarterly data
    df['qoq_growth'] = df.groupby(['ticker', 'metric'])['value'].pct_change(1)
    df['yoy_growth'] = df.groupby(['ticker', 'metric'])['value'].pct_change(4)
    
    # Pivot for easier final consumption: Rows = (ticker, date), Cols = Metrics
    metrics_pivot = df.pivot(index=['ticker', 'reported_date'], columns='metric', values='value').reset_index()
    growth_pivot = df.pivot(index=['ticker', 'reported_date'], columns='metric', values='yoy_growth')
    
    # Rename growth columns
    growth_pivot.columns = [f"{col}_YoY_Growth" for col in growth_pivot.columns]
    growth_pivot = growth_pivot.reset_index()
    
    # Merge values and growth
    final_df = pd.merge(metrics_pivot, growth_pivot, on=['ticker', 'reported_date'])

    final_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Financial growth metrics saved to {OUTPUT_FILE} ({len(final_df)} quarters processed)")

if __name__ == "__main__":
    process_metrics()
