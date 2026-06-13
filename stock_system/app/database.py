import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "stocks.db")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", )


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS historical_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER
        );

        CREATE TABLE IF NOT EXISTS insider_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filing_date TEXT,
            trade_date TEXT,
            insider_name TEXT,
            title TEXT,
            trade_type TEXT,
            price REAL,
            qty INTEGER,
            owned INTEGER,
            delta_own TEXT,
            value TEXT
        );

        CREATE TABLE IF NOT EXISTS options_chain (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            contract_symbol TEXT,
            last_trade_date TEXT,
            strike REAL,
            last_price REAL,
            bid REAL,
            ask REAL,
            volume INTEGER,
            open_interest INTEGER,
            implied_volatility REAL,
            in_the_money INTEGER,
            option_type TEXT
        );

        CREATE TABLE IF NOT EXISTS news_dataset (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_type TEXT,
            title TEXT,
            url TEXT,
            content TEXT
        );
    """)
    conn.commit()
    conn.close()


def _clean_money(val):
    if isinstance(val, str):
        return val.replace("$", "").replace(",", "").replace("-", "").strip()
    return val


def seed_db():
    conn = get_conn()
    cur = conn.cursor()

    # Historical prices
    hist_path = os.path.join(DATA_DIR, "HistoricalData_1780315577803.csv")
    if os.path.exists(hist_path):
        df = pd.read_csv(hist_path)
        df.columns = [c.strip() for c in df.columns]
        for col in ["Close/Last", "Open", "High", "Low"]:
            df[col] = df[col].astype(str).str.replace("$", "", regex=False).str.strip().astype(float)
        for _, row in df.iterrows():
            cur.execute("""
                INSERT OR IGNORE INTO historical_prices (date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (row["Date"], row["Open"], row["High"], row["Low"], row["Close/Last"], row["Volume"]))

    # Insider trades
    insider_path = os.path.join(DATA_DIR, "nvda_insider_trades.csv")
    if os.path.exists(insider_path):
        df = pd.read_csv(insider_path, encoding="utf-8")
        cur.execute("DELETE FROM insider_trades")
        for _, row in df.iterrows():
            try:
                price = float(_clean_money(row.get("Price", 0)) or 0)
                qty_raw = str(row.get("Qty", "0")).replace(",", "").replace("-", "").strip()
                qty = int(float(qty_raw)) if qty_raw else 0
                owned_raw = str(row.get("Owned", "0")).replace(",", "").strip()
                owned = int(float(owned_raw)) if owned_raw else 0
            except:
                price, qty, owned = 0, 0, 0
            cur.execute("""
                INSERT INTO insider_trades (filing_date, trade_date, insider_name, title, trade_type, price, qty, owned, delta_own, value)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(row.get("Filing Date", "")), str(row.get("Trade Date", "")),
                str(row.get("Insider Name", "")), str(row.get("Title", "")),
                str(row.get("Trade Type", "")), price, qty, owned,
                str(row.get("ΔOwn", "")), str(row.get("Value", ""))
            ))

    # Options chain
    options_path = os.path.join(DATA_DIR, "NVDA_Live_Options_Chain.csv")
    if os.path.exists(options_path):
        df = pd.read_csv(options_path)
        cur.execute("DELETE FROM options_chain")
        for _, row in df.iterrows():
            cur.execute("""
                INSERT INTO options_chain (contract_symbol, last_trade_date, strike, last_price, bid, ask, volume, open_interest, implied_volatility, in_the_money, option_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(row.get("contractSymbol", "")), str(row.get("lastTradeDate", "")),
                float(row.get("strike", 0)), float(row.get("lastPrice", 0)),
                float(row.get("bid", 0)), float(row.get("ask", 0)),
                int(0 if pd.isna(row.get("volume")) else row.get("volume", 0)),
                int(0 if pd.isna(row.get("openInterest")) else row.get("openInterest", 0)),
                float(0 if pd.isna(row.get("impliedVolatility")) else row.get("impliedVolatility", 0)),
                int(bool(row.get("inTheMoney", False))),
                str(row.get("Type", ""))
            ))

    # News dataset
    news_path = os.path.join(DATA_DIR, "nvidia_full_dataset.csv")
    if os.path.exists(news_path):
        df = pd.read_csv(news_path)
        cur.execute("DELETE FROM news_dataset")
        for _, row in df.iterrows():
            cur.execute("""
                INSERT INTO news_dataset (source_type, title, url, content)
                VALUES (?, ?, ?, ?)
            """, (
                str(row.get("source_type", "")), str(row.get("title", "")),
                str(row.get("url", "")), str(row.get("content", ""))[:3000]
            ))

    conn.commit()
    conn.close()
