#!/bin/bash

# Script to run the Docker container for Vintern-1B Image-to-Text Demo

# Set image name and container name
IMAGE_NAME="namnhcntt/vintern-image-to-text"
CONTAINER_NAME="vintern-demo"
PORT=8000

# Set default model cache directory
DEFAULT_MODEL_CACHE_DIR="$HOME/.cache/huggingface"
MODEL_CACHE_DIR=${MODEL_CACHE_DIR:-$DEFAULT_MODEL_CACHE_DIR}

# Create model cache directory if it doesn't exist
if [ ! -d "$MODEL_CACHE_DIR" ]; then
    echo "Creating model cache directory: $MODEL_CACHE_DIR"
    mkdir -p "$MODEL_CACHE_DIR"
fi

# Detect architecture and hardware
ARCH=$(uname -m)
OS=$(uname -s)

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    NVIDIA_GPU=1
else
    NVIDIA_GPU=0
fi

# Determine which tag to use
if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
    # macOS on Apple Silicon
    TAG="arm64"
    HARDWARE_TYPE="Apple Silicon"
    DOCKER_RUN_ARGS=""
elif [ $NVIDIA_GPU -eq 1 ]; then
    # System with NVIDIA GPU
    TAG="cuda"
    HARDWARE_TYPE="NVIDIA GPU"
    DOCKER_RUN_ARGS="--gpus all"
else
    # Default to AMD64
    TAG="amd64"
    HARDWARE_TYPE="CPU"
    DOCKER_RUN_ARGS=""
fi

# Check if the specific tag exists
if ! docker image inspect $IMAGE_NAME:$TAG &> /dev/null; then
    echo "Warning: Docker image $IMAGE_NAME:$TAG not found."

    # Check if latest tag exists
    if ! docker image inspect $IMAGE_NAME:latest &> /dev/null; then
        echo "Error: Docker image $IMAGE_NAME:latest not found either."
        echo "Please build the image first using ./build_docker.sh"
        exit 1
    else
        echo "Using $IMAGE_NAME:latest instead."
        TAG="latest"
    fi
fi

echo "Detected hardware: $HARDWARE_TYPE"
echo "Using Docker image: $IMAGE_NAME:$TAG"

# Check if a container with the same name is already running
if docker ps -a --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
    echo "Container $CONTAINER_NAME already exists. Stopping and removing it..."
    docker stop $CONTAINER_NAME &> /dev/null
    docker rm $CONTAINER_NAME &> /dev/null
fi

# Check if port is already in use
if command -v nc &> /dev/null; then
    if nc -z localhost $PORT &> /dev/null; then
        echo "Warning: Port $PORT is already in use."
        echo "Please stop the process using this port or specify a different port."
        exit 1
    fi
elif command -v lsof &> /dev/null; then
    if lsof -i:$PORT &> /dev/null; then
        echo "Warning: Port $PORT is already in use."
        echo "Please stop the process using this port or specify a different port."
        exit 1
    fi
fi

echo "Starting Docker container: $CONTAINER_NAME"
echo "Mapping container port 8000 to host port $PORT"

# Check for environment variables to pass to the container
ENV_VARS=""

# Function to add environment variable if it exists
add_env_var() {
    local var_name=$1
    local env_value=${!var_name}

    if [ -n "$env_value" ]; then
        ENV_VARS="$ENV_VARS -e $var_name=$env_value"
    fi
}

# Add Vintern-specific environment variables if they exist
add_env_var "VINTERN_STREAM"
add_env_var "VINTERN_MAX_TOKENS"
add_env_var "VINTERN_TEMPERATURE"
add_env_var "VINTERN_DO_SAMPLE"
add_env_var "VINTERN_NUM_BEAMS"
add_env_var "VINTERN_REPETITION_PENALTY"
add_env_var "VINTERN_MODEL_ID"

# Prepare volume mapping for model cache
VOLUME_ARGS="-v $MODEL_CACHE_DIR:/root/.cache/huggingface"
echo "Using model cache directory: $MODEL_CACHE_DIR"

# Run the Docker container with appropriate arguments
if [ -n "$DOCKER_RUN_ARGS" ]; then
    echo "Using additional Docker arguments: $DOCKER_RUN_ARGS"
    if [ -n "$ENV_VARS" ]; then
        echo "Using environment variables: $ENV_VARS"
        docker run $DOCKER_RUN_ARGS $ENV_VARS $VOLUME_ARGS --name $CONTAINER_NAME -p $PORT:8000 -d $IMAGE_NAME:$TAG
    else
        docker run $DOCKER_RUN_ARGS $VOLUME_ARGS --name $CONTAINER_NAME -p $PORT:8000 -d $IMAGE_NAME:$TAG
    fi
else
    if [ -n "$ENV_VARS" ]; then
        echo "Using environment variables: $ENV_VARS"
        docker run $ENV_VARS $VOLUME_ARGS --name $CONTAINER_NAME -p $PORT:8000 -d $IMAGE_NAME:$TAG
    else
        docker run $VOLUME_ARGS --name $CONTAINER_NAME -p $PORT:8000 -d $IMAGE_NAME:$TAG
    fi
fi

# Check if container started successfully
if [ $? -eq 0 ]; then
    echo "Container started successfully!"
    echo "API is available at http://localhost:$PORT/api"
    echo "Web interface is available at http://localhost:$PORT"
    echo ""
    echo "To stop the container, use:"
    echo "docker stop $CONTAINER_NAME"
else
    echo "Error: Failed to start container."
    exit 1
fi

# Open the web interface in the default browser
echo "Opening web interface in browser..."
if command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:$PORT
elif command -v open &> /dev/null; then
    open http://localhost:$PORT
elif command -v start &> /dev/null; then
    start http://localhost:$PORT
else
    echo "Could not automatically open the web interface."
    echo "Please open http://localhost:$PORT in your web browser manually."
fi
