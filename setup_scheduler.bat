@echo off
REM Setup Windows Task Scheduler for trading system
REM Run this ONCE as administrator to create all scheduled tasks

set PROJ=C:\Users\liamr\Desktop\spy-trader\.claude\worktrees\flamboyant-lewin

echo Creating task: Start ORB trader on login...
schtasks /create /tn "Trading - ORB Baseline" /tr "pythonw %PROJ%\run_live.py" /sc onlogon /rl highest /f

echo Creating task: Start OpenDrive trader on login...
schtasks /create /tn "Trading - OpenDrive" /tr "pythonw %PROJ%\run_opendrive.py" /sc onlogon /rl highest /f

echo Creating task: Start Pairs trader on login...
schtasks /create /tn "Trading - Pairs" /tr "pythonw %PROJ%\run_pairs.py" /sc onlogon /rl highest /f

echo Creating task: Daily report at 4:05 PM...
schtasks /create /tn "Trading - Daily Report" /tr "python %PROJ%\eod_alert.py" /sc daily /st 16:05 /f

echo.
echo Done! Tasks created:
echo   - Trading - ORB Baseline (starts on login)
echo   - Trading - OpenDrive (starts on login)
echo   - Trading - Pairs (starts on login)
echo   - Trading - Daily Report (runs at 4:05 PM daily)
echo.
echo To remove all tasks later:
echo   schtasks /delete /tn "Trading - ORB Baseline" /f
echo   schtasks /delete /tn "Trading - OpenDrive" /f
echo   schtasks /delete /tn "Trading - Pairs" /f
echo   schtasks /delete /tn "Trading - Daily Report" /f
pause
