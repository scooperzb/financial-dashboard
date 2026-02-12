@echo off
:: ══════════════════════════════════════════════════════
::  Financial Portfolio Dashboard — Launcher
:: ══════════════════════════════════════════════════════
::  Double-click this file to start the dashboard.
::  It will install dependencies (if needed) and launch
::  Streamlit at http://localhost:8501
:: ══════════════════════════════════════════════════════

title Portfolio Dashboard
cd /d "%~dp0"

echo.
echo  ╔══════════════════════════════════════════╗
echo  ║   Financial Portfolio Dashboard          ║
echo  ║   Starting up...                         ║
echo  ╚══════════════════════════════════════════╝
echo.

:: Check Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)

:: Install / update dependencies
echo [1/2] Checking dependencies...
python -m pip install -r requirements.txt --quiet --disable-pip-version-check
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies. Check your internet connection.
    pause
    exit /b 1
)

echo [2/2] Launching dashboard...
echo.
echo  Dashboard will open at: http://localhost:8501
echo  Press Ctrl+C in this window to stop the server.
echo.

python -m streamlit run app.py --server.headless false
pause
