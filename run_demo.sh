#!/bin/bash

# Check if we're in the correct directory
if [ ! -f "app.py" ] || [ ! -f "requirements.txt" ]; then
    echo "Error: This script must be run from the python-demo directory."
    echo "Please cd to the python-demo directory first."
    exit 1
fi

# Check for python and pip availability
PIP_CMD=""
PYTHON_CMD=""

# Find Python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python is not installed or not in PATH."
    echo "Please install Python 3.8+ before running this script."
    exit 1
fi

# Try pip first
if command -v pip &> /dev/null; then
    PIP_CMD="pip"
# Then try pip3
elif command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
# Then try python -m pip
elif $PYTHON_CMD -m pip --version &> /dev/null; then
    PIP_CMD="$PYTHON_CMD -m pip"
else
    echo "Error: pip is not installed or not in PATH."
    echo "Please install pip before running this script."
    exit 1
fi

echo "Vintern-1B Image-to-Text Demo"
echo "=============================="
echo ""
echo "Using Python: $PYTHON_CMD"
echo "Using Pip: $PIP_CMD"
echo "Setting up the environment..."

# Install required packages with upgrade flag to ensure the latest versions
$PIP_CMD install --upgrade -r requirements.txt

# Function to check if a port is in use
check_port() {
    if command -v nc &> /dev/null; then
        nc -z localhost $1 2>/dev/null
        return $?
    elif command -v lsof &> /dev/null; then
        lsof -i:$1 &>/dev/null
        return $?
    else
        # Python fallback
        $PYTHON_CMD -c "
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
$PYTHON_CMD -c "
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