"""
fetch_missing.py
Finds tickers in S&P500_Ticker.csv that have NO entries in the database,
then fetches and saves their transcripts.

Place this file in the same folder as your original script (the project root).
"""

import defeatbeta_api
from defeatbeta_api.data.ticker import Ticker
import sqlite3
import os
import time
import pandas as pd
from datetime import datetime

# ── Paths (same as your original script) ─────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RAW_DIR     = os.path.join(BASE_DIR, 'data', 'raw')
DB_PATH     = os.path.join(BASE_DIR, 'data', 'new_earnings_calls.db')
TICKERS_CSV = os.path.join(BASE_DIR, 'S&P500_Ticker.csv')

os.makedirs(RAW_DIR, exist_ok=True)


def load_csv_tickers():
    df = pd.read_csv(TICKERS_CSV)
    for col in ('Symbol', 'Ticker', 'ticker'):
        if col in df.columns:
            tickers = df[col].tolist()
            break
    else:
        tickers = df.iloc[:, 0].tolist()
    return {str(t).strip().upper() for t in tickers if pd.notna(t) and str(t).strip()}


def load_db_tickers():
    if not os.path.exists(DB_PATH):
        return set()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT ticker FROM transcripts")
    tickers = {row[0].strip().upper() for row in cursor.fetchall()}
    conn.close()
    return tickers


def fetch_and_save(tickers, conn):
    cursor = conn.cursor()
    total = len(tickers)

    for i, ticker_symbol in enumerate(sorted(tickers), 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{total}] Processing {ticker_symbol}...")
        print('='*60)

        try:
            ticker     = Ticker(ticker_symbol)
            transcripts = ticker.earning_call_transcripts()
            tlist       = transcripts.get_transcripts_list()

            if not tlist.empty:
                print(f"  Found {len(tlist)} transcripts")

                for _, row in tlist.iterrows():
                    try:
                        fiscal_year    = int(row['fiscal_year'])
                        fiscal_quarter = int(row['fiscal_quarter'])
                        report_date    = row.get('report_date', None)

                        print(f"  Q{fiscal_quarter} {fiscal_year} ({report_date})...", end=" ")

                        transcript_df  = transcripts.get_transcript(fiscal_year, fiscal_quarter)
                        transcript_text = '\n\n'.join(transcript_df['content'].tolist())

                        filename  = f"{ticker_symbol}_Q{fiscal_quarter}_{fiscal_year}.txt"
                        file_path = os.path.join(RAW_DIR, filename)

                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(transcript_text)

                        cursor.execute('''
                            INSERT OR REPLACE INTO transcripts
                            (ticker, fiscal_year, fiscal_quarter, report_date,
                             transcript_text, file_path, date_collected, char_count)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            ticker_symbol, fiscal_year, fiscal_quarter, report_date,
                            transcript_text, file_path,
                            datetime.now().isoformat(), len(transcript_text)
                        ))
                        conn.commit()

                        print(f"✓ ({len(transcript_text):,} chars)")
                        time.sleep(0.5)

                    except Exception as e:
                        print(f"✗ Error on Q{fiscal_quarter} {fiscal_year}: {e}")
            else:
                print(f"  No transcripts available")

        except Exception as e:
            print(f"✗ Error processing {ticker_symbol}: {e}")

        time.sleep(1)


if __name__ == "__main__":
    csv_tickers = load_csv_tickers()
    db_tickers  = load_db_tickers()

    missing = csv_tickers - db_tickers

    print(f"CSV tickers : {len(csv_tickers)}")
    print(f"DB tickers  : {len(db_tickers)}")
    print(f"Missing     : {len(missing)}")

    if not missing:
        print("\nNothing to do — all CSV tickers are already in the database.")
    else:
        print("\nMissing tickers:")
        for t in sorted(missing):
            print(f"  {t}")

        # ── If you only want to fetch specific tickers, uncomment and edit this:
        missing = {"BRK-B", "ED", "EXPD", "NVR"}

        confirm = input(f"\nFetch transcripts for all {len(missing)} missing tickers? [y/N] ")
        if confirm.strip().lower() == 'y':
            conn = sqlite3.connect(DB_PATH)
            fetch_and_save(missing, conn)
            conn.close()
            print("\n Done!")
        else:
            print("Aborted.")