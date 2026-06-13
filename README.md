# Stock AI Predictor

A multi-asset AI prediction system for stocks, crypto, indices, forex and commodities.  
Predicts next-day price, direction, generates investment signals and trade plans with entry/exit/stop-loss levels.

## Supported Assets
| Type | Examples |
|---|---|
| Stocks | AAPL, NVDA, TSLA, GOOGL, META, MSFT |
| Crypto | BTC-USD, ETH-USD, SOL-USD, XRP-USD |
| Indices | ^NSEI, ^BSESN, ^GSPC, ^IXIC, ^DJI |
| Forex | USDINR=X, EURUSD=X |
| Commodities | GC=F (Gold), CL=F (Oil) |

## Quick Start

### Requirements
- Python 3.10 or higher — download from https://python.org

### Windows
```bat
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
setup.bat        # creates venv + installs all packages
run.bat          # starts the server
```

### Linux / Mac
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
chmod +x setup.sh run.sh
./setup.sh       # creates venv + installs all packages
./run.sh         # starts the server
```

Then open http://127.0.0.1:8000 in your browser.

## Alpha Vantage API Key (optional)
The app uses stooq.com as the primary free data source (no key needed).  
Alpha Vantage is used as a fallback. The default key works for basic use.  
For heavy usage, get a free key at https://www.alphavantage.co/support/#api-key

**Windows:**
```bat
set AV_KEY=your_key_here
run.bat
```

**Linux/Mac:**
```bash
export AV_KEY=your_key_here
./run.sh
```

## Project Structure
```
Stocks/
├── stock_system/
│   ├── app/
│   │   ├── main.py          # FastAPI routes
│   │   ├── ml_model.py      # Model training + prediction + trade plan
│   │   └── database.py      # SQLite setup
│   ├── static/
│   │   └── index.html       # Frontend dashboard
│   ├── models/              # Auto-generated per ticker (gitignored)
│   └── cache/               # Daily data cache (gitignored)
├── run.py                   # Server entry point
├── setup.bat / setup.sh     # One-time setup scripts
├── run.bat / run.sh         # Start scripts
└── requirements.txt         # All Python dependencies
```

## Features
- Next-day price prediction (RandomForest, 23 features)
- 7-day chained forecast
- Investment signal (INVEST / HOLD / WITHDRAW) with scoring
- Trade plan: entry price, stop-loss, target, timeline, quantity tiers
- Technical indicators: RSI, MACD, Bollinger Bands, MA5/20/50
- Fundamentals: Market Cap, P/E, EPS, 52W High/Low
- Same-day data cache — avoids rate limiting
- Auto-retrains on every prediction with fresh live data

## Notes
- No downloads needed after setup — all data is fetched live
- Models and cache are auto-generated and gitignored
- Works on Windows, Linux and Mac
