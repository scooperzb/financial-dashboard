@echo off
:: ══════════════════════════════════════════════════════
::  Portfolio Dashboard — Shared Network Launcher
:: ══════════════════════════════════════════════════════
::  Starts the dashboard accessible to anyone on your
::  local network (Wi-Fi / VPN). Share the URL shown
::  in the console with your teammate.
:: ══════════════════════════════════════════════════════

title Portfolio Dashboard (Shared)
cd /d "%~dp0"

echo.
echo  ╔══════════════════════════════════════════╗
echo  ║   Portfolio Dashboard — Shared Mode      ║
echo  ╚══════════════════════════════════════════╝
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found.
    pause
    exit /b 1
)

echo [1/2] Checking dependencies...
python -m pip install -r requirements.txt --quiet --disable-pip-version-check
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

:: Get local IP address
echo.
echo  ──────────────────────────────────────────
echo  YOUR NETWORK ADDRESS:
echo.
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4 Address"') do (
    for /f "tokens=1" %%b in ("%%a") do (
        echo     http://%%b:8501
    )
)
echo.
echo  Share this URL with your teammate.
echo  They open it in Safari on their iPad.
echo  ──────────────────────────────────────────
echo.
echo  Press Ctrl+C in this window to stop.
echo.

python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
pause
