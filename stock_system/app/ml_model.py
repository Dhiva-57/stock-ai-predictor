import os
import json
import time
import hashlib

import pandas as pd
import numpy as np
import requests

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# Alpha Vantage API key — get a free one at https://www.alphavantage.co/support/#api-key
# Set via environment variable:  set AV_KEY=your_key_here  (Windows)
#                                export AV_KEY=your_key_here (Linux/Mac)
# Or replace the string below directly
AV_KEY = os.environ.get("AV_KEY", "H5ZV6W54WWCNZGTV")
AV_BASE   = "https://www.alphavantage.co/query"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")


def _cache_path(ticker: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe = ticker.upper().replace("-","_").replace("^","").replace("=","_")
    return os.path.join(CACHE_DIR, f"{safe}_{time.strftime('%Y-%m-%d')}.pkl")


def _load_cache(ticker: str):
    """Return cached hist DataFrame if fetched today, else None."""
    path = _cache_path(ticker)
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            pass
    return None


def _save_cache(ticker: str, hist: pd.DataFrame):
    """Save hist to today's cache and clean up yesterday's files."""
    # Clean old cache files for this ticker
    safe = ticker.upper().replace("-","_").replace("^","").replace("=","_")
    for f in os.listdir(CACHE_DIR):
        if f.startswith(safe+"_") and not f.endswith(time.strftime('%Y-%m-%d')+".pkl"):
            try:
                os.remove(os.path.join(CACHE_DIR, f))
            except Exception:
                pass
    joblib.dump(hist, _cache_path(ticker))

FEATURES = [
    "prev_close", "prev_open", "prev_high", "prev_low", "prev_volume",
    "ma5", "ma10", "ma20", "ma50",
    "ema12", "ema26", "macd", "macd_signal",
    "rsi14",
    "bb_upper", "bb_lower", "bb_width",
    "daily_range", "prev_range",
    "pct_change", "volatility5", "volatility10",
    "vol_ratio",
]

# ── Ticker maps for non-standard assets ──────────────────────────────────────
# Alpha Vantage uses different symbols for crypto/forex
_CRYPTO_TICKERS = {
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD",
    "ADA-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "LTC-USD",
}
_FOREX_TICKERS = {
    "USDINR=X", "EURUSD=X", "GBPUSD=X", "USDJPY=X",
}
_INDEX_TICKERS = {
    "^NSEI", "^BSESN", "^GSPC", "^IXIC", "^DJI", "^FTSE", "^N225",
}
# Commodity futures — map to ETF equivalents for Alpha Vantage
_COMMODITY_MAP = {
    "GC=F": "GLD",   # Gold -> Gold ETF
    "CL=F": "USO",   # Oil  -> Oil ETF
    "SI=F": "SLV",   # Silver
    "NG=F": "UNG",   # Natural Gas
}
# Index map to ETFs
_INDEX_MAP = {
    "^GSPC": "SPY",  "^IXIC": "QQQ",  "^DJI": "DIA",
    "^FTSE": "ISF.L","^N225": "EWJ",
    "^NSEI": "INDY", "^BSESN": "INDA",
}


def _av_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0",
    })
    return s


def _resolve_ticker(ticker: str) -> tuple[str, str]:
    """Return (av_symbol, asset_type): 'stock'|'crypto'|'forex'."""
    t = ticker.upper()
    if t in _COMMODITY_MAP:
        return _COMMODITY_MAP[t], "stock"
    if t in _INDEX_MAP:
        return _INDEX_MAP[t], "stock"
    if t in _INDEX_TICKERS:
        return _INDEX_MAP.get(t, t.replace("^", "")), "stock"
    if t in _CRYPTO_TICKERS:
        symbol = t.replace("-USD", "")
        return symbol, "crypto"
    if t in _FOREX_TICKERS:
        pair = t.replace("=X", "")
        return pair, "forex"
    return t, "stock"


def _fetch_stock(symbol: str, session: requests.Session) -> pd.DataFrame:
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol":   symbol,
        "apikey":   AV_KEY,
    }
    r = session.get(AV_BASE, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "Time Series (Daily)" not in data:
        note = data.get("Note") or data.get("Information") or data.get("Error Message", "")
        raise ValueError(f"Alpha Vantage error for {symbol}: {note}")
    ts = data["Time Series (Daily)"]
    rows = []
    for date, v in ts.items():
        rows.append({
            "date":   date,
            "open":   float(v["1. open"]),
            "high":   float(v["2. high"]),
            "low":    float(v["3. low"]),
            "close":  float(v["4. close"]),
            "volume": float(v["5. volume"]),
        })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df.sort_values("date").tail(504)


def _fetch_crypto(symbol: str, session: requests.Session) -> pd.DataFrame:
    params = {
        "function":      "DIGITAL_CURRENCY_DAILY",
        "symbol":        symbol,
        "market":        "USD",
        "apikey":        AV_KEY,
    }
    r = session.get(AV_BASE, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    key = "Time Series (Digital Currency Daily)"
    if key not in data:
        note = data.get("Note") or data.get("Information") or data.get("Error Message", "")
        raise ValueError(f"Alpha Vantage crypto error for {symbol}: {note}")
    ts = data[key]
    rows = []
    for date, v in ts.items():
        rows.append({
            "date":   date,
            "open":   float(v.get("1. open",  v.get("1a. open (USD)", 0))),
            "high":   float(v.get("2. high",  v.get("2a. high (USD)", 0))),
            "low":    float(v.get("3. low",   v.get("3a. low (USD)",  0))),
            "close":  float(v.get("4. close", v.get("4a. close (USD)",0))),
            "volume": float(v.get("5. volume", 0)),
        })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df.sort_values("date").tail(504)


def _fetch_forex(pair: str, session: requests.Session) -> pd.DataFrame:
    from_sym = pair[:3]
    to_sym   = pair[3:]
    params = {
        "function":    "FX_DAILY",
        "from_symbol": from_sym,
        "to_symbol":   to_sym,
        "apikey":      AV_KEY,
    }
    r = session.get(AV_BASE, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    key = "Time Series FX (Daily)"
    if key not in data:
        note = data.get("Note") or data.get("Information") or data.get("Error Message", "")
        raise ValueError(f"Alpha Vantage forex error for {pair}: {note}")
    ts = data[key]
    rows = []
    for date, v in ts.items():
        rows.append({
            "date":   date,
            "open":   float(v["1. open"]),
            "high":   float(v["2. high"]),
            "low":    float(v["3. low"]),
            "close":  float(v["4. close"]),
            "volume": 0.0,
        })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df.sort_values("date").tail(504)


def _fetch_stooq(symbol: str) -> pd.DataFrame:
    """Fetch from stooq.com — free, no key, no rate limit."""
    from io import StringIO
    sym_map = {
        # Crypto
        "BTC-USD": "btc.v",   "ETH-USD": "eth.v",   "SOL-USD": "sol.v",
        "BNB-USD": "bnb.v",   "XRP-USD": "xrp.v",   "DOGE-USD": "doge.v",
        "ADA-USD": "ada.v",   "LTC-USD": "ltc.v",
        # US Indices
        "^GSPC":  "^spx",     "^IXIC":  "^ndx",     "^DJI":  "^dji",
        "^FTSE":  "^ftx",     "^N225":  "^nkx",
        # Indian Indices
        "^NSEI":  "^nsei",    "^BSESN": "^bsesn",
        # Commodities
        "GC=F":   "gc.f",     "CL=F":   "cl.f",
        "SI=F":   "si.f",     "NG=F":   "ng.f",
        # Forex
        "USDINR=X": "inrusd", "EURUSD=X": "eurusd",
        "GBPUSD=X": "gbpusd", "USDJPY=X": "usdjpy",
    }
    # For plain stocks not in map, stooq uses TICKER.US format
    t = symbol.upper()
    if t in sym_map:
        stooq_sym = sym_map[t]
    elif t.startswith("^") or "=" in t or "-" in t:
        stooq_sym = t.lower()
    else:
        stooq_sym = f"{t.lower()}.us"

    url = f"https://stooq.com/q/d/l/?s={stooq_sym}&i=d"
    s   = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    r   = s.get(url, timeout=20)
    r.raise_for_status()
    text = r.text.strip()
    if not text or len(text.splitlines()) < 3 or "No data" in text:
        raise ValueError(f"stooq: no data for {symbol} (tried {stooq_sym})")
    df = pd.read_csv(StringIO(text))
    df.columns = [c.strip().lower() for c in df.columns]
    if "close" not in df.columns:
        raise ValueError(f"stooq: unexpected response for {symbol}")
    if "vol" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"vol": "volume"})
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])
    df = df.sort_values("date").tail(504)
    if len(df) < 30:
        raise ValueError(f"stooq: insufficient rows ({len(df)}) for {symbol}")
    return df


def _fetch_nvda_csv() -> pd.DataFrame:
    csv_path = os.path.join(os.path.dirname(__file__), "..", "..",
                            "HistoricalData_1780315577803.csv")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    for col in ["Close/Last", "Open", "High", "Low"]:
        df[col] = df[col].astype(str).str.replace("$", "", regex=False).astype(float)
    df = df.rename(columns={"Close/Last": "close", "Open": "open",
                             "High": "high",  "Low": "low",
                             "Volume": "volume", "Date": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df[["date", "open", "high", "low", "close", "volume"]].dropna()


def fetch_live_data(ticker: str) -> dict:
    session          = _av_session()
    av_symbol, atype = _resolve_ticker(ticker)
    hist             = None
    last_err         = None

    # Check same-day cache first
    cached = _load_cache(ticker)
    if cached is not None and not cached.empty:
        hist = cached

    # Attempt 1 — stooq
    if hist is None or hist.empty:
        try:
            hist = _fetch_stooq(ticker)
            if hist is not None and not hist.empty:
                _save_cache(ticker, hist)
        except Exception as e:
            last_err = str(e)

    # Attempt 2 — Alpha Vantage
    if hist is None or hist.empty:
        try:
            if atype == "crypto":
                hist = _fetch_crypto(av_symbol, session)
            elif atype == "forex":
                hist = _fetch_forex(av_symbol, session)
            else:
                hist = _fetch_stock(av_symbol, session)
            if hist is not None and not hist.empty:
                _save_cache(ticker, hist)
        except Exception as e:
            last_err = str(e)

    # Attempt 3 — NVDA local CSV
    if (hist is None or hist.empty) and ticker.upper() in ("NVDA", "NVIDIA"):
        try:
            hist = _fetch_nvda_csv()
        except Exception as e:
            last_err = str(e)

    if hist is None or hist.empty:
        raise ValueError(
            f"Could not fetch data for {ticker.upper()}. Error: {last_err}"
        )

    # Build fundamentals — compute what we can from hist, fill rest from OVERVIEW
    last_close  = float(hist["close"].iloc[-1])
    year_ago    = hist[hist["date"] >= (pd.Timestamp(hist["date"].iloc[-1]) - pd.DateOffset(years=1)).strftime("%Y-%m-%d")]
    w52_high    = float(hist["high"].tail(252).max())  if len(hist) >= 5  else None
    w52_low     = float(hist["low"].tail(252).min())   if len(hist) >= 5  else None
    avg_vol     = float(hist["volume"].tail(20).mean()) if len(hist) >= 20 else None

    info = {
        "name":           ticker.upper(),
        "current_price":  round(last_close, 4),
        "week52_high":    round(w52_high, 4) if w52_high else None,
        "week52_low":     round(w52_low,  4) if w52_low  else None,
        "avg_volume":     int(avg_vol)        if avg_vol  else None,
        "market_cap":     None,
        "pe_ratio":       None,
        "eps":            None,
        "beta":           None,
        "sector":         None,
        "industry":       None,
        "dividend_yield": None,
    }

    if atype == "stock":
        try:
            r  = session.get(AV_BASE, params={"function": "OVERVIEW",
                                               "symbol":   av_symbol,
                                               "apikey":   AV_KEY}, timeout=15)
            ov = r.json()
            def _f(key):
                v = ov.get(key, "None")
                return float(v) if v not in ("None", "", None, "-") else None
            if ov.get("Name"):
                info["name"]           = ov["Name"]
            info["market_cap"]     = _f("MarketCapitalization")
            info["pe_ratio"]       = _f("PERatio")
            info["eps"]            = _f("EPS")
            info["beta"]           = _f("Beta")
            info["dividend_yield"] = _f("DividendYield")
            info["sector"]         = ov.get("Sector")   or None
            info["industry"]       = ov.get("Industry") or None
            # Use AV 52w values only if available, otherwise keep hist-computed ones
            if _f("52WeekHigh"): info["week52_high"] = _f("52WeekHigh")
            if _f("52WeekLow"):  info["week52_low"]  = _f("52WeekLow")
        except Exception:
            pass  # Keep hist-computed values

    elif atype == "crypto":
        info["sector"]   = "Cryptocurrency"
        info["industry"] = "Digital Assets"

    elif atype == "forex":
        info["sector"]   = "Forex"
        info["industry"] = "Currency Exchange"

    return {"hist": hist, "info": info, "options": []}


def _rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    n  = len(df)

    df["prev_close"]   = df["close"].shift(1)
    df["prev_open"]    = df["open"].shift(1)
    df["prev_high"]    = df["high"].shift(1)
    df["prev_low"]     = df["low"].shift(1)
    df["prev_volume"]  = df["volume"].shift(1)

    df["ma5"]          = df["close"].rolling(min(5,  n-1), min_periods=1).mean()
    df["ma10"]         = df["close"].rolling(min(10, n-1), min_periods=1).mean()
    df["ma20"]         = df["close"].rolling(min(20, n-1), min_periods=1).mean()
    df["ma50"]         = df["close"].rolling(min(50, n-1), min_periods=1).mean()

    df["ema12"]        = df["close"].ewm(span=12, adjust=False).mean()
    df["ema26"]        = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]         = df["ema12"] - df["ema26"]
    df["macd_signal"]  = df["macd"].ewm(span=9, adjust=False).mean()

    df["rsi14"]        = _rsi(df["close"], min(14, n-2))

    w = min(20, n-1)
    bb_mid             = df["close"].rolling(w, min_periods=1).mean()
    bb_std             = df["close"].rolling(w, min_periods=2).std().fillna(0)
    df["bb_upper"]     = bb_mid + 2 * bb_std
    df["bb_lower"]     = bb_mid - 2 * bb_std
    df["bb_width"]     = df["bb_upper"] - df["bb_lower"]

    df["daily_range"]  = df["high"] - df["low"]
    df["prev_range"]   = df["daily_range"].shift(1)
    df["pct_change"]   = df["close"].pct_change()
    df["volatility5"]  = df["pct_change"].rolling(min(5,  n-1), min_periods=2).std().fillna(0)
    df["volatility10"] = df["pct_change"].rolling(min(10, n-1), min_periods=2).std().fillna(0)
    vol_ma20           = df["volume"].rolling(min(20, n-1), min_periods=1).mean()
    df["vol_ratio"]    = df["volume"] / vol_ma20.replace(0, np.nan)
    df["vol_ratio"]    = df["vol_ratio"].fillna(1.0)

    df["direction"]    = (df["close"] > df["prev_close"]).astype(int)
    return df.dropna(subset=["prev_close", "macd", "rsi14"])


def _next_trading_day(last_date: str) -> str:
    """Return the next calendar day, skipping weekends."""
    d = pd.Timestamp(last_date) + pd.Timedelta(days=1)
    while d.weekday() >= 5:  # 5=Sat, 6=Sun
        d += pd.Timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def _model_paths(ticker: str):
    safe = ticker.upper().replace("-", "_").replace("^", "").replace("=", "_")
    return (
        os.path.join(MODEL_DIR, f"{safe}_price.pkl"),
        os.path.join(MODEL_DIR, f"{safe}_dir.pkl"),
        os.path.join(MODEL_DIR, f"{safe}_scaler.pkl"),
        os.path.join(MODEL_DIR, f"{safe}_metrics.json"),
    )


def train_ticker(ticker: str, hist: pd.DataFrame) -> dict:
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = _build_features(hist)
    if len(df) < 20:
        raise ValueError(f"Not enough data for {ticker} ({len(df)} rows)")

    X, y_price, y_dir = df[FEATURES].values, df["close"].values, df["direction"].values
    X_train, X_test, yp_train, yp_test, yd_train, yd_test = train_test_split(
        X, y_price, y_dir, test_size=0.2, shuffle=False)

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    reg = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    reg.fit(X_train_s, yp_train)
    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf.fit(X_train_s, yd_train)

    mae     = mean_absolute_error(yp_test, reg.predict(X_test_s))
    mae_pct = (mae / np.mean(yp_test)) * 100
    acc     = accuracy_score(yd_test, clf.predict(X_test_s))

    metrics = {
        "ticker": ticker.upper(), "price_mae": round(mae, 4),
        "price_mae_pct": round(mae_pct, 4),
        "direction_accuracy": round(acc * 100, 2),
        "train_samples": int(len(X_train)), "test_samples": int(len(X_test)),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    p_reg, p_clf, p_scaler, p_metrics = _model_paths(ticker)
    joblib.dump(reg, p_reg); joblib.dump(clf, p_clf); joblib.dump(scaler, p_scaler)
    with open(p_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def forecast_ticker(ticker: str, hist: pd.DataFrame, days: int = 7) -> list:
    """Chain predictions forward for `days` trading days."""
    p_reg, p_clf, p_scaler, _ = _model_paths(ticker)
    reg, clf, scaler = joblib.load(p_reg), joblib.load(p_clf), joblib.load(p_scaler)

    # Work on a copy so we can append synthetic rows
    df = _build_features(hist).copy()
    forecast = []

    for i in range(days):
        latest_row = df.iloc[-1]
        feat_vals  = df[FEATURES].iloc[[-1]].values
        feat_s     = scaler.transform(feat_vals)

        pred_price = float(reg.predict(feat_s)[0])
        dir_val    = int(clf.predict(feat_s)[0])
        confidence = float(max(clf.predict_proba(feat_s)[0])) * 100
        last_close = float(latest_row["close"])
        pred_date  = _next_trading_day(str(df["date"].iloc[-1]))

        forecast.append({
            "day":        i + 1,
            "date":       pred_date,
            "price":      round(pred_price, 2),
            "direction":  "UP" if dir_val == 1 else "DOWN",
            "confidence": round(confidence, 2),
            "change":     round(pred_price - last_close, 2),
            "change_pct": round(((pred_price - last_close) / last_close) * 100, 2),
        })

        # Build a synthetic next row using predicted close
        avg_range  = float(latest_row["daily_range"])
        new_open   = last_close
        new_close  = pred_price
        new_high   = pred_price + avg_range * 0.5
        new_low    = pred_price - avg_range * 0.5
        new_volume = float(latest_row["prev_volume"])

        new_row = pd.DataFrame([{
            "date": pred_date, "open": new_open, "high": new_high,
            "low": new_low, "close": new_close, "volume": new_volume
        }])
        hist = pd.concat([hist, new_row], ignore_index=True)
        df   = _build_features(hist)

    return forecast


def _investment_signal(pred_price: float, last_close: float, last_row,
                       forecast: list, confidence: float) -> dict:
    """Generate clear invest/withdraw/hold signals with reasoning."""
    rsi        = float(last_row["rsi14"])
    macd       = float(last_row["macd"])
    macd_sig   = float(last_row["macd_signal"])
    close      = float(last_row["close"])
    ma5        = float(last_row["ma5"])
    ma20       = float(last_row["ma20"])
    bb_upper   = float(last_row["bb_upper"])
    bb_lower   = float(last_row["bb_lower"])
    pct_change = float(last_row["pct_change"])

    score   = 0   # -10 to +10
    reasons = []

    # 1. Model prediction direction
    if pred_price > last_close:
        score += 2
        reasons.append(("bullish", f"Model predicts price UP to ${pred_price:.2f}"))
    else:
        score -= 2
        reasons.append(("bearish", f"Model predicts price DOWN to ${pred_price:.2f}"))

    # 2. Confidence
    if confidence >= 65:
        score += 1
        reasons.append(("bullish", f"High model confidence ({confidence:.1f}%)"))
    elif confidence < 52:
        score -= 1
        reasons.append(("neutral", f"Low model confidence ({confidence:.1f}%) — uncertain"))

    # 3. RSI
    if rsi < 30:
        score += 2
        reasons.append(("bullish", f"RSI {rsi:.1f} — Oversold, strong buy zone"))
    elif rsi > 70:
        score -= 2
        reasons.append(("bearish", f"RSI {rsi:.1f} — Overbought, consider taking profit"))
    elif rsi < 45:
        score += 1
        reasons.append(("bullish", f"RSI {rsi:.1f} — Below midpoint, room to grow"))
    elif rsi > 55:
        score -= 1
        reasons.append(("neutral", f"RSI {rsi:.1f} — Above midpoint, watch for reversal"))

    # 4. MACD crossover
    if macd > macd_sig:
        score += 2
        reasons.append(("bullish", "MACD above signal line — bullish momentum"))
    else:
        score -= 2
        reasons.append(("bearish", "MACD below signal line — bearish momentum"))

    # 5. Price vs Moving Averages
    if close > ma5 > ma20:
        score += 1
        reasons.append(("bullish", "Price above MA5 & MA20 — uptrend confirmed"))
    elif close < ma5 < ma20:
        score -= 1
        reasons.append(("bearish", "Price below MA5 & MA20 — downtrend confirmed"))

    # 6. Bollinger Band position
    if close <= bb_lower:
        score += 1
        reasons.append(("bullish", "Price at/below lower Bollinger Band — potential bounce"))
    elif close >= bb_upper:
        score -= 1
        reasons.append(("bearish", "Price at/above upper Bollinger Band — potential pullback"))

    # 7. 7-day forecast trend
    if len(forecast) >= 3:
        up_days = sum(1 for f in forecast if f["direction"] == "UP")
        if up_days >= 5:
            score += 1
            reasons.append(("bullish", f"7-day forecast: {up_days}/7 days predicted UP"))
        elif up_days <= 2:
            score -= 1
            reasons.append(("bearish", f"7-day forecast: only {up_days}/7 days predicted UP"))

    # Determine action
    if score >= 4:
        action     = "INVEST NOW"
        action_cls = "invest"
        summary    = "Strong bullish signals across multiple indicators. Good time to enter."
    elif score >= 2:
        action     = "CONSIDER BUYING"
        action_cls = "invest-weak"
        summary    = "Moderately bullish. Consider buying on dips or small position."
    elif score <= -4:
        action     = "WITHDRAW / SELL"
        action_cls = "withdraw"
        summary    = "Strong bearish signals. Consider taking profits or cutting losses."
    elif score <= -2:
        action     = "CONSIDER SELLING"
        action_cls = "withdraw-weak"
        summary    = "Moderately bearish. Consider reducing position size."
    else:
        action     = "HOLD"
        action_cls = "hold"
        summary    = "Mixed signals. Hold current position and wait for clearer direction."

    return {
        "action":     action,
        "action_cls": action_cls,
        "score":      score,
        "summary":    summary,
        "reasons":    reasons,
    }


def _trade_plan(ticker: str, last_close: float, pred_price: float,
                forecast: list, last_row, signal: dict,
                confidence: float) -> dict:
    """
    Generate a full trade plan:
    - Entry price & date (when to buy/sell)
    - Exit / target price & date (when to close)
    - Stop-loss price
    - Suggested quantity tiers (conservative / moderate / aggressive)
    - Expected profit/loss per tier
    - Timeline summary
    """
    atr      = float(last_row["daily_range"])          # Average True Range proxy
    vol5     = float(last_row["volatility5"]) or 0.01  # 5-day volatility
    action   = signal["action"]
    is_buy   = "INVEST" in action or "BUYING" in action
    is_sell  = "WITHDRAW" in action or "SELLING" in action

    # ── Entry ──────────────────────────────────────────────────────────────────
    # Buy on a slight dip from current price, sell on a slight bounce
    if is_buy:
        entry_price = round(last_close * 0.998, 2)   # 0.2% below close (limit order)
        entry_note  = "Place a limit buy order slightly below current price"
    elif is_sell:
        entry_price = round(last_close * 1.002, 2)   # 0.2% above close (limit sell)
        entry_note  = "Place a limit sell order slightly above current price"
    else:
        entry_price = round(last_close, 2)
        entry_note  = "Hold — no new position recommended"

    entry_date = _next_trading_day(
        forecast[0]["date"] if forecast else time.strftime("%Y-%m-%d")
    )

    # ── Stop-Loss ─────────────────────────────────────────────────────────────
    # 1.5x ATR below entry for buys, above entry for sells
    sl_distance  = max(atr * 1.5, last_close * 0.02)  # at least 2%
    if is_buy:
        stop_loss    = round(entry_price - sl_distance, 2)
        sl_pct       = round((entry_price - stop_loss) / entry_price * 100, 2)
    elif is_sell:
        stop_loss    = round(entry_price + sl_distance, 2)
        sl_pct       = round((stop_loss - entry_price) / entry_price * 100, 2)
    else:
        stop_loss    = round(last_close * 0.95, 2)
        sl_pct       = 5.0

    # ── Target / Exit ─────────────────────────────────────────────────────────
    # Find the best forecast day to exit (highest price for buys, lowest for sells)
    target_price = pred_price
    target_date  = forecast[0]["date"] if forecast else entry_date
    target_day   = 1

    if forecast:
        if is_buy:
            best = max(forecast, key=lambda x: x["price"])
        elif is_sell:
            best = min(forecast, key=lambda x: x["price"])
        else:
            best = forecast[0]
        target_price = best["price"]
        target_date  = best["date"]
        target_day   = best["day"]

    # Risk / Reward
    if is_buy:
        reward   = round(target_price - entry_price, 2)
        risk     = round(entry_price - stop_loss, 2)
    elif is_sell:
        reward   = round(entry_price - target_price, 2)
        risk     = round(stop_loss - entry_price, 2)
    else:
        reward   = 0.0
        risk     = 0.0

    rr_ratio = round(reward / risk, 2) if risk > 0 else 0

    # ── Quantity Tiers ────────────────────────────────────────────────────────
    # Based on fixed capital amounts: $1000, $5000, $10000
    def qty_tier(capital: float) -> dict:
        if entry_price <= 0:
            return {"capital": capital, "shares": 0, "profit": 0, "loss": 0}
        shares      = int(capital // entry_price)
        profit      = round(shares * reward, 2)
        max_loss    = round(shares * risk, 2)
        return {
            "capital":   capital,
            "shares":    shares,
            "profit_if_target": profit,
            "loss_if_stop":     max_loss,
            "return_pct":       round((profit / capital) * 100, 2) if capital > 0 else 0,
        }

    tiers = {
        "conservative":  qty_tier(1000),
        "moderate":      qty_tier(5000),
        "aggressive":    qty_tier(10000),
    }

    # ── Timeline ──────────────────────────────────────────────────────────────
    if not forecast:
        hold_days = 1
    else:
        hold_days = target_day

    if hold_days <= 1:
        timeline = "Intraday to overnight trade"
    elif hold_days <= 3:
        timeline = f"Short-term swing trade ({hold_days} days)"
    else:
        timeline = f"Multi-day swing trade ({hold_days} days)"

    return {
        "action":       action,
        "is_buy":       is_buy,
        "is_sell":      is_sell,
        "entry_price":  entry_price,
        "entry_date":   entry_date,
        "entry_note":   entry_note,
        "stop_loss":    stop_loss,
        "stop_loss_pct":sl_pct,
        "target_price": round(target_price, 2),
        "target_date":  target_date,
        "target_day":   target_day,
        "reward":       reward,
        "risk":         risk,
        "rr_ratio":     rr_ratio,
        "tiers":        tiers,
        "timeline":     timeline,
        "hold_days":    hold_days,
    }


def predict_ticker(ticker: str, hist: pd.DataFrame, info: dict) -> dict:
    metrics           = train_ticker(ticker, hist)
    p_reg, p_clf, p_scaler, _ = _model_paths(ticker)
    reg, clf, scaler  = joblib.load(p_reg), joblib.load(p_clf), joblib.load(p_scaler)

    df        = _build_features(hist)
    last_row  = df.iloc[-1]
    last_close = float(df["close"].iloc[-1])

    # ── Next day prediction (uses last known close as input) ──
    latest_s   = scaler.transform(df[FEATURES].iloc[[-1]].values)
    pred_price = float(reg.predict(latest_s)[0])
    dir_val    = int(clf.predict(latest_s)[0])
    confidence = float(max(clf.predict_proba(latest_s)[0])) * 100

    # ── Current day prediction (uses second-to-last row as input) ──
    # Simulates what the model would have predicted for today using yesterday's data
    today_pred_price, today_dir, today_conf, today_actual = None, None, None, None
    if len(df) >= 3:
        prev_s            = scaler.transform(df[FEATURES].iloc[[-2]].values)
        today_pred_price  = round(float(reg.predict(prev_s)[0]), 2)
        today_dir_val     = int(clf.predict(prev_s)[0])
        today_conf        = round(float(max(clf.predict_proba(prev_s)[0])) * 100, 2)
        today_dir         = "UP" if today_dir_val == 1 else "DOWN"
        today_actual      = round(last_close, 2)
        today_actual_dir  = "UP" if last_close > float(df["close"].iloc[-2]) else "DOWN"

    # ── 7-day forecast ──
    fc = forecast_ticker(ticker, hist, days=7)

    # ── Investment signal ──
    signal = _investment_signal(pred_price, last_close, last_row, fc, confidence)

    # ── Trade plan ──
    trade_plan = _trade_plan(ticker, last_close, pred_price, fc, last_row, signal, confidence)

    return {
        "ticker": ticker.upper(), "name": info.get("name", ticker.upper()),
        # Next day
        "predicted_price":     round(pred_price, 2),
        "predicted_direction": "UP" if dir_val == 1 else "DOWN",
        "confidence_pct":      round(confidence, 2),
        "last_close":          round(last_close, 2),
        "price_change":        round(pred_price - last_close, 2),
        "price_change_pct":    round(((pred_price - last_close) / last_close) * 100, 2),
        "last_date":           str(df["date"].iloc[-1]),
        "prediction_date":     _next_trading_day(str(df["date"].iloc[-1])),
        # Current day
        "current_day": {
            "date":         str(df["date"].iloc[-1]),
            "pred_price":   today_pred_price,
            "pred_dir":     today_dir,
            "confidence":   today_conf,
            "actual_price": today_actual,
            "actual_dir":   today_actual_dir if len(df) >= 3 else None,
            "correct":      (today_dir == today_actual_dir) if len(df) >= 3 else None,
        } if today_pred_price else None,
        # Signal
        "signal":     signal,
        "trade_plan": trade_plan,
        "forecast":   fc,
        "metrics":  metrics,
        "technicals": {
            "rsi14":       round(float(last_row["rsi14"]), 2),
            "macd":        round(float(last_row["macd"]), 4),
            "macd_signal": round(float(last_row["macd_signal"]), 4),
            "ma5":         round(float(last_row["ma5"]), 2),
            "ma20":        round(float(last_row["ma20"]), 2),
            "ma50":        round(float(last_row["ma50"]), 2),
            "bb_upper":    round(float(last_row["bb_upper"]), 2),
            "bb_lower":    round(float(last_row["bb_lower"]), 2),
            "volatility5": round(float(last_row["volatility5"]) * 100, 4),
        },
        "fundamentals": {
            "current_price":  info.get("current_price"),
            "market_cap":     info.get("market_cap"),
            "pe_ratio":       info.get("pe_ratio"),
            "eps":            info.get("eps"),
            "week52_high":    info.get("week52_high"),
            "week52_low":     info.get("week52_low"),
            "beta":           info.get("beta"),
            "sector":         info.get("sector"),
            "industry":       info.get("industry"),
            "dividend_yield": info.get("dividend_yield"),
        }
    }
