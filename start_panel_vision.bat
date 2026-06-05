@echo off
title Panel Vision - Starting...

REM ── Change to the app folder ──────────────────────────────────
cd /d "%~dp0"

REM ── Activate virtualenv if present, skip if not ──────────────
if exist "venv\Scripts\activate.bat" (
    echo [BOOT] Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo [BOOT] No venv found - using system Python
)

REM ── Wait for network to be ready ─────────────────────────────
echo [BOOT] Waiting for network...
timeout /t 10 /nobreak >nul

REM ── Start the application ────────────────────────────────────
echo [BOOT] Starting Panel Vision...
python app_vision.py

REM ── If app crashes, restart in 5s ────────────────────────────
echo [BOOT] App stopped - restarting in 5s...
timeout /t 5 /nobreak >nul
start "" "%~f0"
