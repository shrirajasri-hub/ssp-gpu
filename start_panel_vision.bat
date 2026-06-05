@echo off
REM ════════════════════════════════════════════════════════════════
REM  Panel Vision Inspection System — Auto-Start on Windows Boot
REM  Vidana Consulting Pvt Ltd
REM
REM  INSTALL (run once as Administrator):
REM    1. Right-click this file → "Create shortcut"
REM    2. Press Win+R → type shell:startup → OK
REM    3. Move the shortcut into that Startup folder
REM    4. Reboot — app starts automatically
REM
REM  OR use Task Scheduler (recommended):
REM    Run setup_task.bat as Administrator
REM ════════════════════════════════════════════════════════════════

title Panel Vision - Starting...

REM ── Change to the app folder ──────────────────────────────────
cd /d "%~dp0"

REM ── Activate virtualenv if present ───────────────────────────
if exist "venv\Scripts\activate.bat" (
    echo [BOOT] Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM ── Wait for network to be ready (important after reboot) ────
echo [BOOT] Waiting for network...
timeout /t 10 /nobreak >nul

REM ── Start the application ────────────────────────────────────
echo [BOOT] Starting Panel Vision Inspection System...
echo [BOOT] Server: http://localhost:5000
echo [BOOT] Cameras will auto-start from config.json

python app_vision.py

REM ── If app crashes, wait 5s and restart ──────────────────────
echo [BOOT] App stopped — restarting in 5s...
timeout /t 5 /nobreak >nul
goto :eof
