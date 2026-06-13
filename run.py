import sys
import os

# Ensure required folders exist
os.makedirs(os.path.join("stock_system", "models"), exist_ok=True)
os.makedirs(os.path.join("stock_system", "cache"),  exist_ok=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stock_system", "app"))

print("=" * 55)
print("  Multi-Stock AI Prediction System - Starting Up")
print("=" * 55)

print("\n[1/2] Initializing database...")
from database import init_db, seed_db, get_conn
init_db()
seed_db()
conn = get_conn()
for table in ["historical_prices", "insider_trades", "options_chain", "news_dataset"]:
    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"      {table}: {count} records")
conn.close()
print("      Database OK")

print("\n[2/2] System ready")
print("      Models train live per ticker on first predict")
print("      Data source: Alpha Vantage API")
print("      NOTE: The demo API key allows only NVDA/IBM/AAPL.")
print("      For all tickers (BTC, NSEI etc.) get a FREE key at:")
print("      https://www.alphavantage.co/support/#api-key")
print("      Then open stock_system/app/ml_model.py and set AV_KEY = 'YOUR_KEY'")
print("\n" + "=" * 55)
print("  Dashboard: http://127.0.0.1:8000")
print("  Supports any stock ticker (NVDA, AAPL, TSLA, etc.)")
print("  Press CTRL+C to stop")
print("=" * 55 + "\n")

import uvicorn
from main import app

uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
