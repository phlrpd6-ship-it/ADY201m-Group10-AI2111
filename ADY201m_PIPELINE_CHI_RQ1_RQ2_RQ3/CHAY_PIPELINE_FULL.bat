@echo off
cd /d "%~dp0"
py ADY201m_RQ1_RQ2_RQ3_PIPELINE.py
if errorlevel 1 echo [LOI] Khong chay duoc. Kiem tra requirements.txt
pause
