"""
check_tickers.py
Compare tickers in a database table against a CSV file to find missing ones.

Usage:
    python check_tickers.py

Configure the DB_CONFIG, TABLE, and CSV settings below before running.
"""

import csv
import sys

# ── Configuration ─────────────────────────────────────────────────────────────

DB_TYPE = "sqlite"   # "postgresql" | "mysql" | "sqlite" | "mssql"

DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,           # 5432 postgres | 3306 mysql | 1433 mssql
    "database": "/home/whitoria/Earning-Calls-Project/data/new_earnings_calls.db",
    "user":     "your_user",
    "password": "your_password",
    # For SQLite, only supply:  "database": "/path/to/file.db"
}

TABLE         = "transcripts"          # table that holds tickers
TICKER_COLUMN = "ticker"          # column name for the ticker symbol

CSV_FILE      = "S&P500_Ticker.csv"     # path to your CSV
CSV_COLUMN    = "Symbol"          # column header in the CSV  (set to None → use first column)

# ── DB connection ─────────────────────────────────────────────────────────────

def get_connection():
    if DB_TYPE == "postgresql":
        import psycopg2
        return psycopg2.connect(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            dbname=DB_CONFIG["database"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
        )
    elif DB_TYPE == "mysql":
        import mysql.connector
        return mysql.connector.connect(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            database=DB_CONFIG["database"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
        )
    elif DB_TYPE == "sqlite":
        import sqlite3
        return sqlite3.connect(DB_CONFIG["database"])
    elif DB_TYPE == "mssql":
        import pyodbc
        conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={DB_CONFIG['host']},{DB_CONFIG['port']};"
            f"DATABASE={DB_CONFIG['database']};"
            f"UID={DB_CONFIG['user']};PWD={DB_CONFIG['password']}"
        )
        return pyodbc.connect(conn_str)
    else:
        raise ValueError(f"Unsupported DB_TYPE: {DB_TYPE}")


# ── Fetch tickers from the database ──────────────────────────────────────────

def get_db_tickers():
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT DISTINCT {TICKER_COLUMN} FROM {TABLE}")
        rows = cursor.fetchall()
        return {row[0].strip().upper() for row in rows if row[0]}
    finally:
        conn.close()


# ── Read tickers from the CSV ─────────────────────────────────────────────────

def get_csv_tickers():
    tickers = set()
    with open(CSV_FILE, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        # If no header column specified, use the first column
        col = CSV_COLUMN or reader.fieldnames[0]

        if col not in reader.fieldnames:
            available = ", ".join(reader.fieldnames)
            raise ValueError(
                f"Column '{col}' not found in CSV.\n"
                f"Available columns: {available}"
            )

        for row in reader:
            val = row[col]
            if val:
                tickers.add(val.strip().upper())

    return tickers


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Connecting to {DB_TYPE} database …")
    try:
        db_tickers = get_db_tickers()
    except Exception as e:
        print(f"[ERROR] Could not read from database: {e}")
        sys.exit(1)

    print(f"Reading CSV: {CSV_FILE} …")
    try:
        csv_tickers = get_csv_tickers()
    except FileNotFoundError:
        print(f"[ERROR] CSV file not found: {CSV_FILE}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Could not read CSV: {e}")
        sys.exit(1)

    # ── Comparison ────────────────────────────────────────────────────────────
    missing_from_db  = csv_tickers - db_tickers   # in CSV but not in DB
    extra_in_db      = db_tickers  - csv_tickers  # in DB but not in CSV

    print(f"\n{'─'*50}")
    print(f"  DB tickers  : {len(db_tickers):>6,}")
    print(f"  CSV tickers : {len(csv_tickers):>6,}")
    print(f"{'─'*50}")
    print(f"  Missing from DB (in CSV, not in DB) : {len(missing_from_db):>6,}")
    print(f"  Extra in DB (in DB, not in CSV)     : {len(extra_in_db):>6,}")
    print(f"{'─'*50}\n")

    if missing_from_db:
        print("TICKERS MISSING FROM DATABASE:")
        for t in sorted(missing_from_db):
            print(f"  {t}")
    else:
        print("No tickers missing from the database.")

    if extra_in_db:
        print("\nTICKERS IN DATABASE BUT NOT IN CSV:")
        for t in sorted(extra_in_db):
            print(f"  {t}")

    # ── Optional: write results to files ─────────────────────────────────────
    if missing_from_db:
        out_file = "missing_tickers.csv"
        with open(out_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ticker"])
            writer.writerows([[t] for t in sorted(missing_from_db)])
        print(f"\nMissing tickers saved to: {out_file}")


if __name__ == "__main__":
    main()