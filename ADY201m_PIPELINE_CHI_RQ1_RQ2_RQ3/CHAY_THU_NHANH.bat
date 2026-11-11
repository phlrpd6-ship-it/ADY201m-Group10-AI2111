@echo off
cd /d "%~dp0"
py ADY201m_RQ1_RQ2_RQ3_PIPELINE.py --quick-check
if errorlevel 1 echo [LOI] Kiem tra Python / thu vien
pause
