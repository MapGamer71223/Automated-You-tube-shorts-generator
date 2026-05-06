@echo off
cd /d "%~dp0"

title [AI SYSTEM] Shorts GUI Launcher
color 0a

echo Initializing environment...
timeout /t 1 >nul

REM 🔥 Fully initialize conda (correct way)
call C:\Users\punya\miniconda3\Scripts\activate.bat

REM 🔥 Activate your environment
call conda activate ytbot

REM 🔒 Prevent global site-packages pollution (VERY IMPORTANT)
set PYTHONNOUSERSITE=1

echo 🔍 Python being used:
where python

echo 🚀 Launching GUI...
python gui.py

pause