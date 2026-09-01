@echo off
echo 🚀 Setting up Multi-Agent Debate Engine Environment for Windows...

python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH. Please install Python 3.10+ first.
    pause
    exit /b 1
)

if not exist "venv" (
    echo 📦 Creating virtual environment 'venv'...
    python -m venv venv
) else (
    echo ✅ Virtual environment 'venv' already exists.
)

echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

echo 📥 Installing dependencies from requirements.txt...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo 🎉 Setup complete!
echo To activate the environment in PowerShell/CMD, run:
echo     venv\Scripts\activate
pause
