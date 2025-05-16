#!/bin/bash

# Script to build the Docker image for Vintern-1B Image-to-Text Demo

# Set image name and tag
IMAGE_NAME="t2nh/vintern-image-to-text"
TAG="latest"

# Check if we're in the correct directory
if [ ! -f "app.py" ]; then
    echo "Error: This script must be run from the python-demo directory."
    echo "Please cd to the python-demo directory first."
    exit 1
fi

# Ensure static directory exists
if [ ! -d "static" ]; then
    echo "Error: static directory not found."
    echo "Please create the static directory and move index.html into it."
    exit 1
fi

# Ensure index.html is in the root directory
if [ ! -f "index.html" ]; then
    echo "Error: index.html not found in root directory."
    echo "Please create index.html in the root directory."
    exit 1
fi

# Create symbolic link from static/index.html to index.html if it doesn't exist
if [ ! -f "static/index.html" ]; then
    echo "Creating symbolic link from static/index.html to index.html..."
    ln -sf ../index.html static/index.html
fi

# Detect architecture
ARCH=$(uname -m)
OS=$(uname -s)

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    NVIDIA_GPU=1
else
    NVIDIA_GPU=0
fi

# Determine which Dockerfile to use
DOCKERFILE=""
PLATFORM=""
HARDWARE_TYPE=""

if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
    # macOS on Apple Silicon
    DOCKERFILE="Dockerfile.arm64"
    PLATFORM="linux/arm64"
    HARDWARE_TYPE="Apple Silicon"
    TAG="arm64"
elif [ $NVIDIA_GPU -eq 1 ]; then
    # System with NVIDIA GPU
    DOCKERFILE="Dockerfile.cuda"
    PLATFORM="linux/amd64"
    HARDWARE_TYPE="NVIDIA GPU"
    TAG="cuda"
else
    # Default to AMD64
    DOCKERFILE="Dockerfile.amd64"
    PLATFORM="linux/amd64"
    HARDWARE_TYPE="CPU"
    TAG="amd64"
fi

# Check if the selected Dockerfile exists
if [ ! -f "$DOCKERFILE" ]; then
    echo "Error: $DOCKERFILE not found."
    echo "Please make sure all Dockerfile variants are available."
    exit 1
fi

echo "Detected hardware: $HARDWARE_TYPE"
echo "Using Dockerfile: $DOCKERFILE"
echo "Building Docker image: $IMAGE_NAME:$TAG"
echo "This may take a few minutes..."

# Build the Docker image
docker build --platform=$PLATFORM -f $DOCKERFILE -t $IMAGE_NAME:$TAG .

# Check if build was successful
if [ $? -eq 0 ]; then
    # Tag as latest as well
    docker tag $IMAGE_NAME:$TAG $IMAGE_NAME:latest

    echo "Docker image built successfully!"
    echo "Created images:"
    echo "  - $IMAGE_NAME:$TAG"
    echo "  - $IMAGE_NAME:latest"
    echo ""
    echo "To run the container with model cache volume mapping, use:"
    echo "docker run -v \$HOME/.cache/huggingface:/root/.cache/huggingface -p 8000:8000 $IMAGE_NAME:$TAG"
    echo ""
    echo "Or use the run_docker.sh script (recommended):"
    echo "./run_docker.sh"
    echo ""
    echo "The run_docker.sh script automatically creates a volume mapping to your local"
    echo "Hugging Face cache directory, preventing model re-downloads on container startup."
else
    echo "Error: Docker build failed."
    exit 1
fi
