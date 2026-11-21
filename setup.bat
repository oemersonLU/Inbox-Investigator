@echo off
cd /d "%~dp0"
echo === Inbox Investigator setup ===
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.10 or newer from python.org and tick "Add Python to PATH".
  pause
  exit /b 1
)
python -m venv .venv
if errorlevel 1 (
  echo Could not create the virtual environment.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
  echo Package install failed. Check your internet connection and try again.
  pause
  exit /b 1
)
echo.
echo Setup complete. Next: put the Kaggle CSV file(s) in data\raw, then double-click train.bat.
pause
