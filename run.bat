@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
  echo The virtual environment is missing. Run setup.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
echo Starting Inbox Investigator. Open http://127.0.0.1:5000 in your browser. Close this window to stop.
python -m inbox_investigator.app
pause
