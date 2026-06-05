@echo off
cd /d "%~dp0"

REM Write all output to a log file you can read after
set LOGFILE=%~dp0startup_log.txt
echo Panel Vision Startup Log > "%LOGFILE%"
echo Time: %date% %time% >> "%LOGFILE%"
echo Folder: %~dp0 >> "%LOGFILE%"

REM Check py
echo Checking py... >> "%LOGFILE%"
py --version >> "%LOGFILE%" 2>&1
if errorlevel 1 (
    echo FAILED: py not found >> "%LOGFILE%"
    echo py not found - check Python install >> "%LOGFILE%"
    goto END
)
echo py OK >> "%LOGFILE%"

REM Check app_vision.py
echo Checking app_vision.py... >> "%LOGFILE%"
if not exist "app_vision.py" (
    echo FAILED: app_vision.py not found in %~dp0 >> "%LOGFILE%"
    goto END
)
echo app_vision.py found >> "%LOGFILE%"

REM Activate venv if exists
if exist "venv\Scripts\activate.bat" (
    echo Activating venv >> "%LOGFILE%"
    call venv\Scripts\activate.bat
) else (
    echo No venv using system Python >> "%LOGFILE%"
)

REM Wait for network
echo Waiting for network >> "%LOGFILE%"
timeout /t 10 /nobreak >nul

REM Start app
echo Starting app_vision.py >> "%LOGFILE%"
py app_vision.py >> "%LOGFILE%" 2>&1

echo App stopped >> "%LOGFILE%"

:END
echo. >> "%LOGFILE%"
echo Done >> "%LOGFILE%"

REM Open the log file so you can read it
notepad "%LOGFILE%"
