#!/bin/bash

# Script to build multi-architecture Docker images for Vintern-1B Image-to-Text Demo

# Set image name and tag
IMAGE_NAME="vietprogrammer/vintern-image-to-text"
VERSION="1.1.0"
TAG="latest"

# Check if we're in the correct directory
if [ ! -f "app.py" ]; then
    echo "Error: This script must be run from the project root directory."
    echo "Please cd to the project root directory first."
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

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH."
    echo "Please install Docker before running this script."
    exit 1
fi

# Check if Docker Buildx is available
if ! docker buildx version &> /dev/null; then
    echo "Error: Docker Buildx is not available."
    echo "Please install Docker Buildx or update Docker to a newer version."
    exit 1
fi

# Create a new builder instance if it doesn't exist
if ! docker buildx inspect multiarch-builder &> /dev/null; then
    echo "Creating a new Docker Buildx builder instance..."
    # Thêm tùy chọn --driver để sử dụng driver docker-container với socket mặc định
    docker buildx create --name multiarch-builder --driver docker-container --use
else
    echo "Using existing Docker Buildx builder instance..."
    # Xóa builder cũ nếu có lỗi và tạo lại
    if ! docker buildx use multiarch-builder &> /dev/null; then
        echo "Existing builder instance may be corrupted. Removing and recreating..."
        docker buildx rm multiarch-builder &> /dev/null
        docker buildx create --name multiarch-builder --driver docker-container --use
    else
        docker buildx use multiarch-builder
    fi
fi

# Tăng thời gian chờ bootstrap
echo "Bootstrapping Docker Buildx builder instance (this may take a moment)..."
DOCKER_BUILDKIT=1 docker buildx inspect --bootstrap

# Parse command line arguments
BUILD_ARM64=false
BUILD_AMD64=false
BUILD_CUDA=false
BUILD_ALL=false
PUSH=false

# If no arguments, show help
if [ $# -eq 0 ]; then
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --arm64    Build for ARM64 architecture (Apple Silicon, etc.)"
    echo "  --amd64    Build for AMD64 architecture (Intel/AMD CPUs)"
    echo "  --cuda     Build for NVIDIA CUDA"
    echo "  --all      Build for all architectures"
    echo "  --push     Push images to Docker Hub (requires login)"
    echo ""
    echo "Example: $0 --all --push"
    exit 0
fi

# Parse arguments
for arg in "$@"; do
    case $arg in
        --arm64)
            BUILD_ARM64=true
            ;;
        --amd64)
            BUILD_AMD64=true
            ;;
        --cuda)
            BUILD_CUDA=true
            ;;
        --all)
            BUILD_ALL=true
            ;;
        --push)
            PUSH=true
            ;;
        *)
            echo "Unknown option: $arg"
            exit 1
            ;;
    esac
done

# If --all is specified, build all architectures
if [ "$BUILD_ALL" = true ]; then
    BUILD_ARM64=true
    BUILD_AMD64=true
    BUILD_CUDA=true
fi

# Check if at least one architecture is selected
if [ "$BUILD_ARM64" = false ] && [ "$BUILD_AMD64" = false ] && [ "$BUILD_CUDA" = false ]; then
    echo "Error: No architecture selected."
    echo "Please specify at least one architecture to build."
    exit 1
fi

# If push is requested, check if user is logged in to Docker Hub
if [ "$PUSH" = true ]; then
    echo "Checking Docker Hub login status..."
    if ! docker info | grep -q "Username"; then
        echo "Error: You are not logged in to Docker Hub."
        echo "Please run 'docker login' before using the --push option."
        exit 1
    fi
fi

# Build for ARM64
if [ "$BUILD_ARM64" = true ]; then
    echo "Building Docker image for ARM64 architecture..."

    # Build the ARM64 image
    if [ "$PUSH" = true ]; then
        docker buildx build --platform linux/arm64 -f Dockerfile.arm64 -t ${IMAGE_NAME}:arm64 -t ${IMAGE_NAME}:${VERSION}-arm64 --push .
    else
        docker buildx build --platform linux/arm64 -f Dockerfile.arm64 -t ${IMAGE_NAME}:arm64 -t ${IMAGE_NAME}:${VERSION}-arm64 --load .
    fi

    # Check if build was successful
    if [ $? -eq 0 ]; then
        echo "ARM64 image built successfully: ${IMAGE_NAME}:arm64"
    else
        echo "Error: ARM64 image build failed."
        exit 1
    fi
fi

# Build for AMD64
if [ "$BUILD_AMD64" = true ]; then
    echo "Building Docker image for AMD64 architecture..."

    # Build the AMD64 image
    if [ "$PUSH" = true ]; then
        docker buildx build --platform linux/amd64 -f Dockerfile.amd64 -t ${IMAGE_NAME}:amd64 -t ${IMAGE_NAME}:${VERSION}-amd64 --push .
    else
        docker buildx build --platform linux/amd64 -f Dockerfile.amd64 -t ${IMAGE_NAME}:amd64 -t ${IMAGE_NAME}:${VERSION}-amd64 --load .
    fi

    # Check if build was successful
    if [ $? -eq 0 ]; then
        echo "AMD64 image built successfully: ${IMAGE_NAME}:amd64"
    else
        echo "Error: AMD64 image build failed."
        exit 1
    fi
fi

# Build for CUDA
if [ "$BUILD_CUDA" = true ]; then
    echo "Building Docker image for NVIDIA CUDA..."

    # Build the CUDA image
    if [ "$PUSH" = true ]; then
        docker buildx build --platform linux/amd64 -f Dockerfile.cuda -t ${IMAGE_NAME}:cuda -t ${IMAGE_NAME}:${VERSION}-cuda --push .
    else
        docker buildx build --platform linux/amd64 -f Dockerfile.cuda -t ${IMAGE_NAME}:cuda -t ${IMAGE_NAME}:${VERSION}-cuda --load .
    fi

    # Check if build was successful
    if [ $? -eq 0 ]; then
        echo "CUDA image built successfully: ${IMAGE_NAME}:cuda"
    else
        echo "Error: CUDA image build failed."
        exit 1
    fi
fi

# Create a multi-architecture manifest if building for multiple architectures
if [ "$PUSH" = true ] && { [ "$BUILD_ARM64" = true ] && [ "$BUILD_AMD64" = true ]; } || [ "$BUILD_ALL" = true ]; then
    echo "Creating multi-architecture manifest..."

    # Create and push the manifest for latest
    docker buildx imagetools create -t ${IMAGE_NAME}:latest \
        ${IMAGE_NAME}:arm64 \
        ${IMAGE_NAME}:amd64

    # Create and push the manifest for version
    docker buildx imagetools create -t ${IMAGE_NAME}:${VERSION} \
        ${IMAGE_NAME}:${VERSION}-arm64 \
        ${IMAGE_NAME}:${VERSION}-amd64

    echo "Multi-architecture manifest created: ${IMAGE_NAME}:latest"
    echo "Multi-architecture manifest created: ${IMAGE_NAME}:${VERSION}"
fi

echo ""
echo "Build process completed successfully!"
echo ""
echo "To run the container with model cache volume mapping, use:"
echo "docker run -v \$HOME/.cache/huggingface:/root/.cache/huggingface -p 8000:8000 ${IMAGE_NAME}:<tag>"
echo ""
echo "Or use the run_docker.sh script (recommended):"
echo "./run_docker.sh"
echo ""
echo "Available tags:"
if [ "$BUILD_ARM64" = true ]; then
    echo "  - ${IMAGE_NAME}:arm64 (for Apple Silicon and other ARM64 devices)"
    echo "  - ${IMAGE_NAME}:${VERSION}-arm64 (versioned ARM64 image)"
fi
if [ "$BUILD_AMD64" = true ]; then
    echo "  - ${IMAGE_NAME}:amd64 (for Intel/AMD CPUs)"
    echo "  - ${IMAGE_NAME}:${VERSION}-amd64 (versioned AMD64 image)"
fi
if [ "$BUILD_CUDA" = true ]; then
    echo "  - ${IMAGE_NAME}:cuda (for NVIDIA GPUs)"
    echo "  - ${IMAGE_NAME}:${VERSION}-cuda (versioned CUDA image)"
fi
if [ "$PUSH" = true ] && { [ "$BUILD_ARM64" = true ] && [ "$BUILD_AMD64" = true ]; } || [ "$BUILD_ALL" = true ]; then
    echo "  - ${IMAGE_NAME}:latest (multi-architecture image)"
    echo "  - ${IMAGE_NAME}:${VERSION} (versioned multi-architecture image)"
fi
