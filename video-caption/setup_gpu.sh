#!/bin/bash

# GPU Video Caption Service Setup Script
# This script helps set up and deploy the GPU-enabled video captioning service

set -e  # Exit on any error

echo "=== GPU Video Caption Service Setup ==="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if NVIDIA Docker is available
check_nvidia_docker() {
    print_status "Checking NVIDIA Docker support..."
    
    if ! command -v nvidia-docker &> /dev/null; then
        print_warning "nvidia-docker command not found. Checking for native Docker GPU support..."
        
        if ! docker info | grep -q "nvidia"; then
            print_error "NVIDIA Docker support not detected!"
            echo "Please install NVIDIA Docker runtime or NVIDIA Container Toolkit"
            echo "Visit: https://github.com/NVIDIA/nvidia-docker"
            exit 1
        fi
    fi
    
    print_status "NVIDIA Docker support detected!"
}

# Check NVIDIA GPU availability
check_gpu() {
    print_status "Checking GPU availability..."
    
    if ! command -v nvidia-smi &> /dev/null; then
        print_error "nvidia-smi not found! Please install NVIDIA drivers."
        exit 1
    fi
    
    # Check if GPU is available
    if ! nvidia-smi &> /dev/null; then
        print_error "No NVIDIA GPU detected or drivers not working!"
        exit 1
    fi
    
    print_status "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader,nounits
}

# Build Docker image
build_image() {
    print_status "Building GPU-enabled Docker image..."
    
    # Check if Dockerfile exists
    if [ ! -f "services/video_caption/Dockerfile" ]; then
        print_error "Dockerfile not found at services/video_caption/Dockerfile"
        exit 1
    fi
    
    # Build the image
    docker build -t video-caption-gpu:latest -f services/video_caption/Dockerfile .
    
    if [ $? -eq 0 ]; then
        print_status "Docker image built successfully!"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Test GPU access in container
test_gpu_access() {
    print_status "Testing GPU access in container..."
    
    docker run --rm --gpus all video-caption-gpu:latest python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
else:
    print('GPU not accessible in container!')
    exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_status "GPU access test passed!"
    else
        print_error "GPU access test failed!"
        exit 1
    fi
}

# Start the service
start_service() {
    print_status "Starting GPU-enabled video caption service..."
    
    # Create necessary directories
    mkdir -p /tmp/caption_uploads /tmp/caption_processed
    
    # Use docker-compose if available, otherwise use docker run
    if command -v docker-compose &> /dev/null && [ -f "docker-compose.yml" ]; then
        print_status "Using docker-compose to start service..."
        docker-compose up -d video-caption-gpu
    else
        print_status "Using docker run to start service..."
        docker run -d \
            --name video-caption-gpu \
            --gpus all \
            -p 5003:5003 \
            -v /tmp/caption_uploads:/tmp/caption_uploads \
            -v /tmp/caption_processed:/tmp/caption_processed \
            -e NVIDIA_VISIBLE_DEVICES=all \
            -e CUDA_VISIBLE_DEVICES=0 \
            -e WHISPER_MODEL_SIZE=small \
            --restart unless-stopped \
            video-caption-gpu:latest
    fi
    
    print_status "Service started! Waiting for it to be ready..."
    
    # Wait for service to be ready
    for i in {1..30}; do
        if curl -f http://localhost:5003/health &> /dev/null; then
            print_status "Service is ready!"
            break
        fi
        echo "Waiting... ($i/30)"
        sleep 2
    done
    
    # Check service status
    print_status "Service status:"
    curl -s http://localhost:5003/gpu-status | python -m json.tool
}

# Stop the service
stop_service() {
    print_status "Stopping video caption service..."
    
    if command -v docker-compose &> /dev/null && [ -f "docker-compose.yml" ]; then
        docker-compose down
    else
        docker stop video-caption-gpu || true
        docker rm video-caption-gpu || true
    fi
    
    print_status "Service stopped!"
}

# Show service logs
show_logs() {
    print_status "Showing service logs..."
    
    if command -v docker-compose &> /dev/null && [ -f "docker-compose.yml" ]; then
        docker-compose logs -f video-caption-gpu
    else
        docker logs -f video-caption-gpu
    fi
}

# Main function
main() {
    case "${1:-setup}" in
        "setup")
            check_gpu
            check_nvidia_docker
            build_image
            test_gpu_access
            start_service
            ;;
        "build")
            build_image
            ;;
        "start")
            start_service
            ;;
        "stop")
            stop_service
            ;;
        "restart")
            stop_service
            start_service
            ;;
        "logs")
            show_logs
            ;;
        "test")
            test_gpu_access
            ;;
        "status")
            print_status "Service status:"
            curl -s http://localhost:5003/health | python -m json.tool
            echo
            curl -s http://localhost:5003/gpu-status | python -m json.tool
            ;;
        *)
            echo "Usage: $0 {setup|build|start|stop|restart|logs|test|status}"
            echo
            echo "Commands:"
            echo "  setup   - Complete setup (build, test, start)"
            echo "  build   - Build Docker image"
            echo "  start   - Start the service"
            echo "  stop    - Stop the service"
            echo "  restart - Restart the service"
            echo "  logs    - Show service logs"
            echo "  test    - Test GPU access"
            echo "  status  - Show service status"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"