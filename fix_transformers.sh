#!/bin/bash

echo "Fixing dependencies for Vintern-1B-v3.5 on Apple Silicon..."

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

# Check if running on Apple Silicon
if [[ $(uname -m) == "arm64" && $(uname) == "Darwin" ]]; then
    echo "Detected Apple Silicon (M1/M2/M3) Mac"
    echo "Installing PyTorch and TorchVision optimized for Apple Silicon..."

    # Install PyTorch with MPS support for Apple Silicon
    $PIP_CMD install --upgrade torch==2.1.2 torchvision==0.16.2

    # Verify MPS is available
    $PYTHON_CMD -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"

    # Add note about MPS fallback
    echo ""
    echo "Note: Some PyTorch operations are not yet implemented for MPS."
    echo "The run_demo.sh script will automatically set PYTORCH_ENABLE_MPS_FALLBACK=1"
    echo "to use CPU as fallback for unsupported operations."
    echo ""

    # Create or update .zshrc or .bash_profile to include the MPS fallback variable
    SHELL_PROFILE=""
    if [ -f "$HOME/.zshrc" ]; then
        SHELL_PROFILE="$HOME/.zshrc"
    elif [ -f "$HOME/.bash_profile" ]; then
        SHELL_PROFILE="$HOME/.bash_profile"
    elif [ -f "$HOME/.bashrc" ]; then
        SHELL_PROFILE="$HOME/.bashrc"
    fi

    if [ -n "$SHELL_PROFILE" ]; then
        if ! grep -q "PYTORCH_ENABLE_MPS_FALLBACK" "$SHELL_PROFILE"; then
            echo "# PyTorch MPS fallback for Apple Silicon" >> "$SHELL_PROFILE"
            echo "export PYTORCH_ENABLE_MPS_FALLBACK=1" >> "$SHELL_PROFILE"
            echo "Added PYTORCH_ENABLE_MPS_FALLBACK=1 to $SHELL_PROFILE"
            echo "Please restart your terminal or run 'source $SHELL_PROFILE' to apply changes"
        else
            echo "PYTORCH_ENABLE_MPS_FALLBACK is already set in $SHELL_PROFILE"
        fi
    else
        echo "Could not find shell profile file. Please manually add the following line to your shell profile:"
        echo "export PYTORCH_ENABLE_MPS_FALLBACK=1"
    fi
else
    echo "Not running on Apple Silicon, using standard PyTorch installation"
fi

echo "Installing transformers 4.38.0 or newer..."

# Install the newer transformers version
$PIP_CMD install --upgrade "transformers>=4.38.0"

echo "Done! Now you can run ./run_demo.sh to start the demo."