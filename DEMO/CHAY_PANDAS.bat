@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
py demo_pandas.py
if errorlevel 1 (
    echo.
    echo Chay khong thanh cong. Cai thu vien: py -m pip install -r requirements.txt
)
pause
