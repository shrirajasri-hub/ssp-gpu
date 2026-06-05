@echo off
title Panel Vision

REM ── Go to the folder where this bat file lives ────────────────
cd /d "%~dp0"
echo [INFO] App folder: %~dp0

REM ── Check py is available ─────────────────────────────────────
py --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ============================================
    echo  ERROR: py not found
    echo  Install Python from https://python.org
    echo ============================================
    pause
    exit /b 1
)
echo [INFO] Python found:
py --version

REM ── Check app_vision.py exists ────────────────────────────────
if not exist "app_vision.py" (
    echo.
    echo ============================================
    echo  ERROR: app_vision.py not found
    echo  Move this bat file into the same folder
    echo  as app_vision.py
    echo  Current folder: %~dp0
    echo ============================================
    pause
    exit /b 1
)
echo [INFO] app_vision.py found

REM ── Activate venv if present ──────────────────────────────────
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating venv...
    call venv\Scripts\activate.bat
) else (
    echo [INFO] No venv - using system Python
)

REM ── Wait for network ──────────────────────────────────────────
echo [INFO] Waiting 10s for network...
timeout /t 10 /nobreak >nul

REM ── Start the app ─────────────────────────────────────────────
echo [INFO] Starting Panel Vision...
echo [INFO] Open browser at http://localhost:5000
echo.
py app_vision.py

REM ── If we reach here the app stopped ─────────────────────────
echo.
echo ============================================
echo  App stopped. See error above.
echo ============================================
pause
