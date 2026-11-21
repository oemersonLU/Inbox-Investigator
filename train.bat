@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
  echo The virtual environment is missing. Run setup.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
echo === Training ===
python scripts\train.py
if errorlevel 1 (
  echo Training failed. Read the message above.
  pause
  exit /b 1
)
echo.
echo === Evaluating on the held-out test split ===
python scripts\evaluate.py
if errorlevel 1 (
  echo Evaluation failed. Read the message above.
  pause
  exit /b 1
)
echo.
echo Done. The table for your report is in results\results_table.md
pause
