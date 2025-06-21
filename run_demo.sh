#!/bin/bash

# Check if we're in the correct directory
if [ ! -f "app.py" ] || [ ! -f "requirements.txt" ]; then
    echo "Error: This script must be run from the project directory."
    echo "Please cd to the project directory first."
    exit 1
fi

# Virtual environment setup
VENV_DIR="venv"
VENV_PYTHON=""
VENV_PIP=""

echo "Vintern-1B Image-to-Text Demo"
echo "=============================="
echo ""

# Check if virtual environment exists, if not create it
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found. Creating virtual environment..."
    
    # Find Python command for creating venv
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
    
    # Create virtual environment
    $PYTHON_CMD -m venv $VENV_DIR
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        echo "Please ensure you have the venv module installed."
        exit 1
    fi
    echo "Virtual environment created successfully."
fi

# Activate virtual environment and set commands
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    # Windows (Git Bash, MSYS2, Cygwin)
    source "$VENV_DIR/Scripts/activate"
    VENV_PYTHON="$VENV_DIR/Scripts/python"
    VENV_PIP="$VENV_DIR/Scripts/pip"
else
    # Unix-like systems (Linux, macOS)
    source "$VENV_DIR/bin/activate"
    VENV_PYTHON="$VENV_DIR/bin/python"
    VENV_PIP="$VENV_DIR/bin/pip"
fi

echo "Using virtual environment: $VENV_DIR"
echo "Using Python: $VENV_PYTHON"
echo "Using Pip: $VENV_PIP"

# Run hardware detection first
echo "Detecting hardware..."
HARDWARE_INFO=$($VENV_PYTHON detect_hardware.py --json 2>/dev/null || echo '{"best":"cpu"}')
HARDWARE_TYPE=$(echo "$HARDWARE_INFO" | $VENV_PYTHON -c "import sys, json; data=json.load(sys.stdin); print(data.get('best', 'cpu'))" 2>/dev/null || echo "cpu")

echo "Detected hardware type: $HARDWARE_TYPE"

# Set environment variables based on hardware
if [ "$HARDWARE_TYPE" = "nvidia" ]; then
    echo "NVIDIA GPU detected, will install PyTorch with CUDA support"
    export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
elif [ "$HARDWARE_TYPE" = "apple_silicon" ]; then
    echo "Apple Silicon detected, enabling MPS fallback"
    export PYTORCH_ENABLE_MPS_FALLBACK=1
fi

echo "Setting up the environment..."

# Install the correct PyTorch version based on detected hardware
if [ "$HARDWARE_TYPE" = "nvidia" ]; then
    echo "Installing PyTorch with CUDA support..."
    # Uninstall any existing PyTorch to avoid conflicts
    $VENV_PIP uninstall -y torch torchvision 2>/dev/null || true
    # Install PyTorch with CUDA support from PyTorch's official index
    $VENV_PIP install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    echo "PyTorch with CUDA support installed"
elif [ "$HARDWARE_TYPE" = "apple_silicon" ]; then
    echo "Installing PyTorch optimized for Apple Silicon..."
    # For Apple Silicon, ensure we have the right version
    $VENV_PIP install torch torchvision
    echo "PyTorch for Apple Silicon installed"
else
    echo "Installing PyTorch for CPU..."
    $VENV_PIP install torch torchvision
    echo "PyTorch for CPU installed"
fi

# Install other requirements (excluding torch and torchvision to avoid overwriting)
echo "Installing other dependencies..."
# Create temporary requirements file without torch and torchvision
grep -v "^torch" requirements.txt > temp_requirements.txt
$VENV_PIP install --upgrade -r temp_requirements.txt
rm temp_requirements.txt

# Verify PyTorch device availability
echo ""
echo "Verifying PyTorch device availability..."
$VENV_PYTHON -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device count: {torch.cuda.device_count()}')
    print(f'CUDA device name: {torch.cuda.get_device_name(0)}')
    print(f'CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

# Check for MPS (Apple Silicon)
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print(f'MPS available: True')
    import platform
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        print('Apple Silicon MPS will be used')

# Show what device will be used
import sys, os
sys.path.append('.')
try:
    from app import get_best_device_and_dtype
    device, dtype, device_name = get_best_device_and_dtype()
    print(f'App will use: {device_name} ({device}) with dtype {dtype}')
except Exception as e:
    print(f'Note: Could not load device detection from app.py: {e}')
"
echo ""

# Function to check if a port is in use
check_port() {
    if command -v nc &> /dev/null; then
        nc -z localhost $1 2>/dev/null
        return $?
    elif command -v lsof &> /dev/null; then
        lsof -i:$1 &>/dev/null
        return $?
    else
        # Python fallback using virtual environment python
        $VENV_PYTHON -c "
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.connect(('localhost', $1))
    s.close()
    exit(0)
except:
    exit(1)
"
        return $?
    fi
}

# Check for running processes on port 8000
PORT=8000
while check_port $PORT; do
    echo "Port $PORT is already in use, trying next port..."
    PORT=$((PORT+1))
    if [ $PORT -gt 8020 ]; then
        echo "Error: Could not find an available port in range 8000-8020."
        echo "Please manually stop the process using port 8000 and try again."
        exit 1
    fi
done

echo "Using port $PORT for the API server."

# Define cleanup function
cleanup() {
    if [ -f "index.html.bak" ]; then
        echo "Restoring original index.html file..."
        mv index.html.bak index.html
    fi

    if [ -n "$SERVER_PID" ]; then
        echo "Stopping server (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null || true
    fi

    # Deactivate virtual environment
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        deactivate
    fi

    echo "Server stopped."
    exit 0
}

# Update the port in the HTML file temporarily
if [ $PORT -ne 8000 ]; then
    echo "Updating API URL in index.html to use port $PORT..."
    sed -i.bak "s/localhost:8000/localhost:$PORT/g" index.html
fi

# Set trap for cleanup
trap cleanup INT EXIT

# Launch the server with environment variables to suppress warnings
echo "Starting the API server..."
export PYTHONWARNINGS="ignore"
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Enable MPS fallback for Apple Silicon
if [[ $(uname -m) == "arm64" && $(uname) == "Darwin" ]]; then
    echo "Detected Apple Silicon - enabling MPS fallback for unsupported operations"
    export PYTORCH_ENABLE_MPS_FALLBACK=1
fi
$VENV_PYTHON -c "
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

import warnings
warnings.filterwarnings('ignore')

import app
app.run_server(port=$PORT)
" &
SERVER_PID=$!

echo "Server started with PID: $SERVER_PID"
echo "API is available at http://localhost:$PORT"
echo ""

# Wait a bit for the server to start
sleep 3

# Attempt to open the HTML file in a browser
echo "Opening the web interface..."
if command -v xdg-open &> /dev/null; then
    xdg-open index.html
elif command -v open &> /dev/null; then
    open index.html
elif command -v start &> /dev/null; then
    start index.html
else
    echo "Could not automatically open the web interface."
    echo "Please open 'index.html' in your web browser manually."
fi

echo ""
echo "Press Ctrl+C to stop the server when you're done."

# Wait for user to press Ctrl+C
wait $SERVER_PID