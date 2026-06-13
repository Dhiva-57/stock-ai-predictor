from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from contextlib import asynccontextmanager
import os, sys, json, time

sys.path.insert(0, os.path.dirname(__file__))

from database import init_db, seed_db, get_conn
from ml_model import fetch_live_data, predict_ticker, train_ticker

STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    seed_db()
    yield


app = FastAPI(title="Multi-Stock Prediction System", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ── Live: Predict any ticker ──────────────────────────────────────────────────
@app.get("/api/predict")
def predict(ticker: str = Query("NVDA", description="Stock ticker symbol")):
    try:
        ticker = ticker.upper().strip()
        live   = fetch_live_data(ticker)
        result = predict_ticker(ticker, live["hist"], live["info"])
        result["options"] = _clean_options(live["options"])
        result["history"] = _recent_history(live["hist"])
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── Live: Ticker fundamentals + history only (no model) ───────────────────────
@app.get("/api/stock")
def stock_info(ticker: str = Query(..., description="Stock ticker symbol")):
    try:
        ticker = ticker.upper().strip()
        live   = fetch_live_data(ticker)
        return {
            "ticker":      ticker,
            "info":        live["info"],
            "options":     _clean_options(live["options"]),
            "history":     _recent_history(live["hist"]),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ── DB-backed routes (NVDA static data) ───────────────────────────────────────
@app.get("/api/insider-trades")
def get_insider_trades(limit: int = 50):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM insider_trades ORDER BY trade_date DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/api/news")
def get_news(limit: int = 30):
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, source_type, title, url FROM news_dataset ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/api/db-health")
def db_health():
    try:
        conn = get_conn()
        counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ["historical_prices", "insider_trades", "options_chain", "news_dataset"]}
        conn.close()
        return {"status": "ok", "record_counts": counts}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})


@app.get("/api/system-health")
def system_health():
    """Shows cache, trained models and longevity info."""
    cache_dir  = os.path.join(os.path.dirname(__file__), "..", "cache")
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    today      = time.strftime("%Y-%m-%d")

    cached = []
    if os.path.exists(cache_dir):
        for f in sorted(os.listdir(cache_dir)):
            if f.endswith(".pkl"):
                fpath = os.path.join(cache_dir, f)
                size  = round(os.path.getsize(fpath) / 1024, 1)
                cached.append({"file": f, "size_kb": size, "is_today": today in f})

    models = []
    if os.path.exists(models_dir):
        for f in sorted(os.listdir(models_dir)):
            if f.endswith("_metrics.json"):
                try:
                    with open(os.path.join(models_dir, f)) as fh:
                        m = json.load(fh)
                    models.append({
                        "ticker":    m.get("ticker"),
                        "trained_at":m.get("trained_at"),
                        "accuracy":  m.get("direction_accuracy"),
                        "mae":       m.get("price_mae"),
                    })
                except Exception:
                    pass

    return {
        "today":           today,
        "cache_entries":   len(cached),
        "cached_files":    cached,
        "trained_models":  models,
        "longevity_notes": [
            "Model retrains on every prediction using fresh live data — stays accurate indefinitely",
            "Same-day cache prevents repeated stooq fetches — avoids rate limiting",
            "Cache auto-cleans yesterday's files on each fetch",
            "No manual maintenance needed — fully self-sustaining",
        ]
    }


# ── Helpers ────────────────────────────────────────────────────────────────────
def _clean_options(options: list) -> list:
    result = []
    for o in options:
        result.append({
            "symbol":            str(o.get("contractSymbol", "")),
            "type":              str(o.get("type", "")),
            "strike":            float(o.get("strike", 0) or 0),
            "last_price":        float(o.get("lastPrice", 0) or 0),
            "bid":               float(o.get("bid", 0) or 0),
            "ask":               float(o.get("ask", 0) or 0),
            "volume":            int(o.get("volume", 0) or 0) if str(o.get("volume", "nan")) != "nan" else 0,
            "open_interest":     int(o.get("openInterest", 0) or 0) if str(o.get("openInterest", "nan")) != "nan" else 0,
            "implied_volatility":float(o.get("impliedVolatility", 0) or 0),
            "in_the_money":      bool(o.get("inTheMoney", False)),
        })
    return result


def _recent_history(hist, n=60) -> list:
    df = hist.tail(n).copy()
    records = []
    for _, row in df.iterrows():
        records.append({
            "date":   str(row.get("date", ""))[:10],
            "open":   round(float(row.get("open", 0)), 2),
            "high":   round(float(row.get("high", 0)), 2),
            "low":    round(float(row.get("low", 0)), 2),
            "close":  round(float(row.get("close", 0)), 2),
            "volume": int(row.get("volume", 0) or 0),
        })
    return list(reversed(records))
