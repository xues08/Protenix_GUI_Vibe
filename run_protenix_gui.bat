@echo off
echo ================================================
echo     Protenix GUI Launcher (Windows)       
echo ================================================

:: Check if Python is installed
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo Error: Python is not installed or not added to your PATH.
    echo Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

echo Checking and installing required Python packages...
python -m pip install PyQt6 PyQt6-WebEngine send2trash

echo.
echo Starting Protenix GUI...
python Protenix_GUI.py

pause
