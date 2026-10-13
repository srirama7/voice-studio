@echo off
REM Voice Studio Enterprise - one-click launcher with auto-restart.
REM Automatically launches the server and opens your browser to http://127.0.0.1:7860
title Voice Studio Enterprise Server
cd /d "C:\Users\amogh\my-voice-studio"

REM Resolve Python interpreter
set "PYTHON_EXE=python"
where python >nul 2>&1
if %errorlevel% neq 0 (
    if exist "C:\Program Files\Blender Foundation\Blender 5.2\5.2\python\bin\python.exe" (
        set "PYTHON_EXE=C:\Program Files\Blender Foundation\Blender 5.2\5.2\python\bin\python.exe"
    )
)

echo [%date% %time%] Launching Voice Studio in your browser...
start "" "http://127.0.0.1:7860"

:loop
echo [%date% %time%] Starting Voice Studio Server on http://127.0.0.1:7860 ...
"%PYTHON_EXE%" "C:\Users\amogh\my-voice-studio\server.py"
echo [%date% %time%] Server stopped (code %errorlevel%). Restarting in 5 seconds...
timeout /t 5 /nobreak >nul
goto loop
