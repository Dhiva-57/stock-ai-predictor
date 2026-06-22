import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "stocks.db")


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


def seed_db():
    """No-op: static CSV files removed. DB tables created by init_db()."""
    pass
