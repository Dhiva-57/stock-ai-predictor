@echo off
echo Fixing yfinance SSL issue on Windows...
C:\Users\User\AppData\Local\Programs\Python\Python311\python.exe -m pip install --upgrade yfinance
C:\Users\User\AppData\Local\Programs\Python\Python311\python.exe -m pip uninstall curl_cffi -y
echo Done. Now run: C:\Users\User\AppData\Local\Programs\Python\Python311\python.exe run.py
pause
