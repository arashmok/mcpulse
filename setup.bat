@echo off
REM Setup script for MCPulse on Windows

echo 🔌 MCPulse Setup Script
echo =======================
echo.

REM Check Python
echo Checking Python version...
python --version
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8 or higher.
    exit /b 1
)
echo ✓ Python found
echo.

REM Create virtual environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo ✓ Virtual environment created
) else (
    echo ✓ Virtual environment already exists
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo ✓ Virtual environment activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1
echo ✓ pip upgraded
echo.

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
echo ✓ Dependencies installed
echo.

REM Create .env file
if not exist ".env" (
    echo Creating .env file from template...
    copy .env.example .env
    echo ✓ .env file created
    echo.
    echo ⚠️  Please edit .env file and add your API keys:
    echo    - OPENAI_API_KEY or ANTHROPIC_API_KEY
    echo    - MongoDB settings ^(if using MongoDB^)
) else (
    echo ✓ .env file already exists
)
echo.

REM Create config directory
if not exist "config" mkdir config
echo ✓ Config directory ready
echo.

echo ✅ Setup complete!
echo.
echo To start the application:
echo   1. Activate virtual environment: venv\Scripts\activate.bat
echo   2. Edit .env file with your API keys
echo   3. Run: python main.py
echo.
echo The application will be available at http://localhost:7860

pause
