import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "stock_system", "app"))

print("=" * 50)
print("STEP 1: DATABASE TEST")
print("=" * 50)

from database import init_db, seed_db, get_conn

init_db()
print("[OK] DB initialized")

seed_db()
print("[OK] DB seeded")

conn = get_conn()
tables = ["historical_prices", "insider_trades", "options_chain", "news_dataset"]
for t in tables:
    count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"  {t}: {count} records")
conn.close()

print()
print("=" * 50)
print("STEP 2: ML MODEL TRAINING & ACCURACY")
print("=" * 50)

from ml_model import train, predict, get_metrics

metrics = train()
print("[OK] Model trained")
print(f"  Direction Accuracy : {metrics['direction_accuracy']}%")
print(f"  Price MAE          : ${metrics['price_mae']}")
print(f"  Price MAE %        : {metrics['price_mae_pct']}%")
print(f"  Train samples      : {metrics['train_samples']}")
print(f"  Test samples       : {metrics['test_samples']}")

print()
print("=" * 50)
print("STEP 3: PREDICTION")
print("=" * 50)

pred = predict()
print(f"  Predicted Price    : ${pred['predicted_price']}")
print(f"  Direction          : {pred['predicted_direction']}")
print(f"  Confidence         : {pred['confidence_pct']}%")
print(f"  Last Known Close   : ${pred['last_known_close']}")
print(f"  Last Date          : {pred['last_date']}")

print()
print("=" * 50)
print("STEP 4: FASTAPI ROUTES CHECK")
print("=" * 50)

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

routes_to_test = [
    ("/api/db-health", "GET"),
    ("/api/prices", "GET"),
    ("/api/insider-trades", "GET"),
    ("/api/options", "GET"),
    ("/api/news", "GET"),
    ("/api/predict", "GET"),
    ("/api/model-metrics", "GET"),
    ("/", "GET"),
]

for path, method in routes_to_test:
    resp = client.get(path)
    status = "[OK]" if resp.status_code == 200 else f"[FAIL {resp.status_code}]"
    print(f"  {status} {method} {path}")

print()
print("All checks complete.")
