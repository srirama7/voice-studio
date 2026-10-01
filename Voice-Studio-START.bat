@echo off
REM Voice Studio Enterprise - one-click launcher with auto-restart.
REM Keeps the website on http://127.0.0.1:7860. If the server ever stops,
REM this window restarts it after 5 seconds. Close this window to stop it.
title Voice Studio Server (do not close while using the website)
cd /d "C:\Users\amogh\my-voice-studio"
:loop
echo [%date% %time%] Starting Voice Studio on http://127.0.0.1:7860 ...
python "C:\Users\amogh\my-voice-studio\app.py"
echo [%date% %time%] Server stopped (code %errorlevel%). Restarting in 5 seconds...
timeout /t 5 /nobreak >nul
goto loop
