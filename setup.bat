@echo off
echo =============================================
echo  Stock AI Predictor - Setup
echo =============================================

echo.
echo [1/3] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo.
echo [2/3] Installing dependencies...
venv\Scripts\pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo [3/3] Creating required folders...
if not exist stock_system\models mkdir stock_system\models
if not exist stock_system\cache  mkdir stock_system\cache

echo.
echo =============================================
echo  Setup complete!
echo  Run the app with:  run.bat
echo =============================================
pause
