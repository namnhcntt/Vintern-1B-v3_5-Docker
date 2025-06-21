#!/bin/bash

echo "Setting up Virtual Environment for Vintern-1B Project"
echo "===================================================="
echo ""

# Check if we're in the correct directory
if [ ! -f "app.py" ] || [ ! -f "requirements.txt" ]; then
    echo "Error: This script must be run from the project directory."
    echo "Please cd to the project directory first."
    exit 1
fi

# Find Python command
PYTHON_CMD=""
if command -v python &> /dev/null; then
    # Check if it's a real Python installation (not Windows Store version)
    if python -c "import sys; print(sys.executable)" 2>/dev/null | grep -v "WindowsApps" > /dev/null; then
        PYTHON_CMD="python"
    fi
fi

if [ -z "$PYTHON_CMD" ] && command -v python3 &> /dev/null; then
    # Check if it's a real Python installation
    if python3 -c "import sys; print(sys.executable)" 2>/dev/null | grep -v "WindowsApps" > /dev/null; then
        PYTHON_CMD="python3"
    fi
fi

if [ -z "$PYTHON_CMD" ]; then
    echo "Error: Python is not installed or not in PATH."
    echo "Please install Python 3.8+ before running this script."
    exit 1
fi

echo "Using Python: $PYTHON_CMD"
echo ""

# Check if virtual environment already exists
VENV_DIR="venv"
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at '$VENV_DIR'."
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        echo "Using existing virtual environment."
        exit 0
    fi
fi

# Create virtual environment
echo "Creating virtual environment..."
$PYTHON_CMD -m venv $VENV_DIR

if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment."
    echo "Please ensure you have the venv module installed."
    exit 1
fi

echo "Virtual environment created successfully at '$VENV_DIR'."
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    # Windows (Git Bash, MSYS2, Cygwin)
    source "$VENV_DIR/Scripts/activate"
    VENV_PIP="$VENV_DIR/Scripts/pip"
else
    # Unix-like systems (Linux, macOS)
    source "$VENV_DIR/bin/activate"
    VENV_PIP="$VENV_DIR/bin/pip"
fi

# Upgrade pip
echo "Upgrading pip..."
$VENV_PIP install --upgrade pip

# Install requirements
echo "Installing requirements..."
$VENV_PIP install -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "Setup completed successfully!"
    echo "Virtual environment is ready at '$VENV_DIR'."
    echo ""
    echo "To run the demo, use: ./run_demo.sh"
    echo "The script will automatically use the virtual environment."
else
    echo ""
    echo "Error: Failed to install requirements."
    echo "Please check the error messages above and try again."
    exit 1
fi
