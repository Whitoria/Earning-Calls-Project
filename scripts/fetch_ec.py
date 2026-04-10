import defeatbeta_api
from defeatbeta_api.data.ticker import Ticker
import sqlite3
import os
import time
import pandas as pd
from datetime import datetime

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DB_PATH = os.path.join(BASE_DIR, 'data', 'new_earnings_calls.db')
TICKERS_CSV = os.path.join(BASE_DIR, 'S&P500_Ticker.csv')  # Path to your CSV

# Create directories if they don't exist
os.makedirs(RAW_DIR, exist_ok=True)

def load_tickers_from_csv(csv_path):
    """Load tickers from CSV file"""
    df = pd.read_csv(csv_path)
    
    # Check what column has the tickers
    if 'Symbol' in df.columns:
        tickers = df['Symbol'].tolist()
    elif 'Ticker' in df.columns:
        tickers = df['Ticker'].tolist()
    elif 'ticker' in df.columns:
        tickers = df['ticker'].tolist()
    else:
        # If first column has tickers
        tickers = df.iloc[:, 0].tolist()
    
    # Clean tickers (remove any NaN or empty strings)
    tickers = [str(t).strip() for t in tickers if pd.notna(t) and str(t).strip()]
    
    print(f"Loaded {len(tickers)} tickers from CSV")
    return tickers

def setup_database():
    """Create database and tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS transcripts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        fiscal_year INTEGER NOT NULL,
        fiscal_quarter INTEGER NOT NULL,
        report_date TEXT,
        transcript_text TEXT,
        file_path TEXT,
        date_collected TEXT,
        char_count INTEGER,
        UNIQUE(ticker, fiscal_year, fiscal_quarter)
    )
    ''')
    
    cursor.execute('''
    CREATE INDEX IF NOT EXISTS idx_ticker_year_quarter 
    ON transcripts(ticker, fiscal_year, fiscal_quarter)
    ''')
    
    conn.commit()
    return conn

def fetch_and_save_transcripts(tickers, conn):
    """Fetch transcripts and save to both files and database"""
    cursor = conn.cursor()
    
    total_tickers = len(tickers)
    
    for i, ticker_symbol in enumerate(tickers, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{total_tickers}] Processing {ticker_symbol}...")
        print('='*60)
        
        try:
            ticker = Ticker(ticker_symbol)
            transcripts = ticker.earning_call_transcripts()
            transcript_list = transcripts.get_transcripts_list()
            
            if not transcript_list.empty:
                print(f"Found {len(transcript_list)} transcripts")
                
                for idx, row in transcript_list.iterrows():
                    try:
                        # Get fiscal year and quarter
                        fiscal_year = int(row['fiscal_year'])
                        fiscal_quarter = int(row['fiscal_quarter'])
                        report_date = row['report_date'] if 'report_date' in row else None
                        
                        print(f"  Q{fiscal_quarter} {fiscal_year} ({report_date})...", end=" ")
                        
                        # Get transcript DataFrame
                        transcript_df = transcripts.get_transcript(fiscal_year, fiscal_quarter)
                        
                        # Convert DataFrame to text (content only, no speaker names)
                        transcript_text = '\n\n'.join(transcript_df['content'].tolist())
                        
                        # Save to file
                        filename = f"{ticker_symbol}_Q{fiscal_quarter}_{fiscal_year}.txt"
                        file_path = os.path.join(RAW_DIR, filename)
                        
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(transcript_text)
                        
                        # Save to database
                        cursor.execute('''
                        INSERT OR REPLACE INTO transcripts 
                        (ticker, fiscal_year, fiscal_quarter, report_date, transcript_text, 
                         file_path, date_collected, char_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            ticker_symbol,
                            fiscal_year,
                            fiscal_quarter,
                            report_date,
                            transcript_text,
                            file_path,
                            datetime.now().isoformat(),
                            len(transcript_text)
                        ))
                        conn.commit()
                        
                        print(f"✓ ({len(transcript_text):,} chars)")
                        time.sleep(0.5)
                        
                    except Exception as e:                      print(f"✗ Error fetching transcript: {e}")
                        
            else:
                print(f"No transcripts available for {ticker_symbol}")
                
        except Exception as e:
            print(f"✗ Error processing {ticker_symbol}: {e}")
        
        time.sleep(1)

def print_summary(conn):
    """Print summary statistics"""
    cursor = conn.cursor()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Count by ticker
    cursor.execute('''
    SELECT ticker, COUNT(*) as count, SUM(char_count) as total_chars
    FROM transcripts
    GROUP BY ticker
    ORDER BY ticker
    ''')
    
    results = cursor.fetchall()
    total_transcripts = 0
    
    for ticker, count, total_chars in results:
        print(f"{ticker:6} | {count:2} transcripts | {total_chars:,} total characters")
        total_transcripts += count
    
    print("="*60)
    print(f"TOTAL: {total_transcripts} transcripts")
    print(f"Database: {DB_PATH}")
    print(f"Raw files: {RAW_DIR}")
    print("="*60)

if __name__ == "__main__":
    print("Starting transcript collection...")
    print(f"Database path: {DB_PATH}")
    print(f"Raw files path: {RAW_DIR}")
    
    # Load tickers from CSV
    tickers = load_tickers_from_csv(TICKERS_CSV)
    
    # Setup database
    conn = setup_database()
    print("✓ Database setup complete")
    
    # Fetch and save transcripts
    fetch_and_save_transcripts(tickers, conn)
    
    # Print summary
    print_summary(conn)
    
    conn.close()
    print("\n✓ All done!")