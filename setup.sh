#!/bin/bash
echo "============================================="
echo " Stock AI Predictor - Setup"
echo "============================================="

echo ""
echo "[1/3] Creating virtual environment..."
python3 -m venv venv

echo ""
echo "[2/3] Installing dependencies..."
venv/bin/pip install -r requirements.txt

echo ""
echo "[3/3] Creating required folders..."
mkdir -p stock_system/models
mkdir -p stock_system/cache

echo ""
echo "============================================="
echo " Setup complete! Run the app with:  ./run.sh"
echo "============================================="
