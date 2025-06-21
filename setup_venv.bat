@echo off
echo Setting up Virtual Environment for Vintern-1B Project
echo ====================================================
echo.

REM Check if we're in the correct directory
if not exist "app.py" (
    echo Error: This script must be run from the project directory.
    echo Please cd to the project directory first.
    pause
    exit /b 1
)

if not exist "requirements.txt" (
    echo Error: requirements.txt not found in current directory.
    echo Please ensure you're in the correct project directory.
    pause
    exit /b 1
)

REM Find Python command
set PYTHON_CMD=
where python >nul 2>&1
if %errorlevel% == 0 (
    set PYTHON_CMD=python
) else (
    where python3 >nul 2>&1
    if %errorlevel% == 0 (
        set PYTHON_CMD=python3
    ) else (
        echo Error: Python is not installed or not in PATH.
        echo Please install Python 3.8+ before running this script.
        pause
        exit /b 1
    )
)

echo Using Python: %PYTHON_CMD%
echo.

REM Check if virtual environment already exists
set VENV_DIR=venv
if exist "%VENV_DIR%" (
    echo Virtual environment already exists at '%VENV_DIR%'.
    set /p "RECREATE=Do you want to recreate it? (y/N): "
    if /i "%RECREATE%"=="y" (
        echo Removing existing virtual environment...
        rmdir /s /q "%VENV_DIR%"
    ) else (
        echo Using existing virtual environment.
        pause
        exit /b 0
    )
)

REM Create virtual environment
echo Creating virtual environment...
%PYTHON_CMD% -m venv %VENV_DIR%

if %errorlevel% neq 0 (
    echo Error: Failed to create virtual environment.
    echo Please ensure you have the venv module installed.
    pause
    exit /b 1
)

echo Virtual environment created successfully at '%VENV_DIR%'.
echo.

REM Activate virtual environment and install requirements
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

REM Upgrade pip
echo Upgrading pip...
"%VENV_DIR%\Scripts\pip.exe" install --upgrade pip

REM Install requirements
echo Installing requirements...
"%VENV_DIR%\Scripts\pip.exe" install -r requirements.txt

if %errorlevel% == 0 (
    echo.
    echo Setup completed successfully!
    echo Virtual environment is ready at '%VENV_DIR%'.
    echo.
    echo To run the demo, use: run_demo.sh
    echo The script will automatically use the virtual environment.
) else (
    echo.
    echo Error: Failed to install requirements.
    echo Please check the error messages above and try again.
)

pause
