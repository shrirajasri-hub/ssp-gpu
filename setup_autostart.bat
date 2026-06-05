@echo off
REM ════════════════════════════════════════════════════════════════
REM  Panel Vision — Windows Task Scheduler Setup
REM  Run ONCE as Administrator.
REM  Task runs at USER LOGIN (not SYSTEM) so browser opens correctly.
REM ════════════════════════════════════════════════════════════════

echo [SETUP] Installing Panel Vision auto-start...

set "APP_DIR=%~dp0"
set "APP_DIR=%APP_DIR:~0,-1%"

if exist "%APP_DIR%\venv\Scripts\python.exe" (
    set "PYTHON=%APP_DIR%\venv\Scripts\python.exe"
) else (
    for /f "delims=" %%i in ('where python 2^>nul') do (
        set "PYTHON=%%i"
        goto :found_python
    )
    echo [SETUP] ERROR: python not found. Install Python first.
    pause & exit /b 1
    :found_python
)

set "SCRIPT=%APP_DIR%\app_vision.py"
set "STARTBAT=%APP_DIR%\start_panel_vision.bat"

echo [SETUP] App folder : %APP_DIR%
echo [SETUP] Python     : %PYTHON%
echo [SETUP] Script     : %SCRIPT%

REM Remove old task
schtasks /delete /tn "PanelVision" /f >nul 2>&1

REM Create XML for the task (ON LOGON of current user)
set "XML=%TEMP%\panel_vision_task.xml"

echo ^<?xml version="1.0" encoding="UTF-16"?^>                            > "%XML%"
echo ^<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task"^> >> "%XML%"
echo   ^<Triggers^>                                                          >> "%XML%"
echo     ^<LogonTrigger^>                                                    >> "%XML%"
echo       ^<Enabled^>true^</Enabled^>                                       >> "%XML%"
echo       ^<Delay^>PT15S^</Delay^>                                          >> "%XML%"
echo     ^</LogonTrigger^>                                                   >> "%XML%"
echo   ^</Triggers^>                                                         >> "%XML%"
echo   ^<Principals^>                                                        >> "%XML%"
echo     ^<Principal id="Author"^>                                           >> "%XML%"
echo       ^<LogonType^>InteractiveToken^</LogonType^>                       >> "%XML%"
echo       ^<RunLevel^>HighestAvailable^</RunLevel^>                         >> "%XML%"
echo     ^</Principal^>                                                      >> "%XML%"
echo   ^</Principals^>                                                       >> "%XML%"
echo   ^<Settings^>                                                          >> "%XML%"
echo     ^<MultipleInstancesPolicy^>IgnoreNew^</MultipleInstancesPolicy^>   >> "%XML%"
echo     ^<DisallowStartIfOnBatteries^>false^</DisallowStartIfOnBatteries^> >> "%XML%"
echo     ^<StopIfGoingOnBatteries^>false^</StopIfGoingOnBatteries^>         >> "%XML%"
echo     ^<RestartOnFailure^>                                                >> "%XML%"
echo       ^<Interval^>PT30S^</Interval^>                                   >> "%XML%"
echo       ^<Count^>3^</Count^>                                              >> "%XML%"
echo     ^</RestartOnFailure^>                                               >> "%XML%"
echo   ^</Settings^>                                                         >> "%XML%"
echo   ^<Actions Context="Author"^>                                          >> "%XML%"
echo     ^<Exec^>                                                            >> "%XML%"
echo       ^<Command^>"%STARTBAT%"^</Command^>                               >> "%XML%"
echo       ^<WorkingDirectory^>%APP_DIR%^</WorkingDirectory^>                >> "%XML%"
echo     ^</Exec^>                                                           >> "%XML%"
echo   ^</Actions^>                                                          >> "%XML%"
echo ^</Task^>                                                               >> "%XML%"

schtasks /create /tn "PanelVision" /xml "%XML%" /f

if %errorlevel% == 0 (
    echo.
    echo ════════════════════════════════════════════════════
    echo  [SETUP] SUCCESS
    echo  Panel Vision will start automatically on next login.
    echo  URL: http://localhost:5000
    echo  Cameras: auto-started from config.json
    echo ════════════════════════════════════════════════════
    echo.
    echo  To enable Windows auto-login (no password on boot):
    echo    Press Win+R → netplwiz → uncheck password
    echo.
    echo  To remove auto-start:
    echo    schtasks /delete /tn "PanelVision" /f
    echo ════════════════════════════════════════════════════
) else (
    echo [SETUP] FAILED — run as Administrator
)

del "%XML%" >nul 2>&1
pause
