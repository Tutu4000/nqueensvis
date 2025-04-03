@echo off
echo Installing required dependencies...
pip install -r requirements.txt

echo.
echo Starting N-Queens Genetic Algorithm Executor...
python executor.py

pause 