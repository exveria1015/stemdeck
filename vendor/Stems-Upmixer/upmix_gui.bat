@echo off
cd /d "%~dp0"
python upmix_cli.py --gui
if errorlevel 1 pause
