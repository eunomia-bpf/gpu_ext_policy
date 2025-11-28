#!/bin/bash
#
# test_gpu_preempt.sh - Test script for gpu_preempt_ctrl
#
# This script helps test the GPU preempt control functionality.
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== GPU Preempt Control Test ===${NC}"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}Warning: Not running as root. Some tests may fail.${NC}"
    echo "Consider running: sudo $0"
    echo ""
fi

# Check for NVIDIA driver with tracepoints
check_tracepoints() {
    echo "1. Checking NVIDIA tracepoints..."
    if [ -d /sys/kernel/tracing/events/nvidia ]; then
        echo -e "   ${GREEN}Found /sys/kernel/tracing/events/nvidia${NC}"
        ls /sys/kernel/tracing/events/nvidia/ 2>/dev/null || true
    else
        echo -e "   ${RED}NVIDIA tracepoints not found!${NC}"
        echo "   Make sure the modified NVIDIA driver is loaded."
        echo "   Expected tracepoints:"
        echo "     - nvidia_gpu_tsg_create"
        echo "     - nvidia_gpu_tsg_schedule"
        echo "     - nvidia_gpu_tsg_destroy"
        return 1
    fi
    echo ""
}

# Build test CUDA program
build_cuda_test() {
    echo "2. Building CUDA test program..."
    if command -v nvcc &> /dev/null; then
        nvcc -o test_preempt_ctrl test_preempt_ctrl.cu 2>&1 || {
            echo -e "   ${RED}Failed to build CUDA test program${NC}"
            return 1
        }
        echo -e "   ${GREEN}Built test_preempt_ctrl${NC}"
    else
        echo -e "   ${YELLOW}nvcc not found, skipping CUDA test build${NC}"
    fi
    echo ""
}

# Build gpu_preempt_ctrl
build_tool() {
    echo "3. Building gpu_preempt_ctrl..."
    make gpu_preempt_ctrl 2>&1 || {
        echo -e "   ${RED}Failed to build gpu_preempt_ctrl${NC}"
        return 1
    }
    echo -e "   ${GREEN}Built gpu_preempt_ctrl${NC}"
    echo ""
}

# Run quick test
run_quick_test() {
    echo "4. Running quick test..."
    echo "   Starting gpu_preempt_ctrl in background..."

    # Run gpu_preempt_ctrl for 5 seconds
    timeout 5 sudo ./gpu_preempt_ctrl -v 2>&1 &
    CTRL_PID=$!

    sleep 1

    if [ -x ./test_preempt_ctrl ]; then
        echo "   Running CUDA test..."
        ./test_preempt_ctrl 2 1000000 &
        CUDA_PID=$!
        sleep 3
        kill $CUDA_PID 2>/dev/null || true
    else
        echo "   Skipping CUDA test (not built)"
    fi

    wait $CTRL_PID 2>/dev/null || true
    echo ""
}

# Print usage
usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  check     - Check for NVIDIA tracepoints"
    echo "  build     - Build the tools"
    echo "  test      - Run quick test"
    echo "  monitor   - Start gpu_preempt_ctrl in verbose mode"
    echo "  cuda      - Run CUDA test program"
    echo "  all       - Run all steps"
    echo "  help      - Show this help"
    echo ""
    echo "Interactive testing:"
    echo "  1. Terminal 1: sudo ./gpu_preempt_ctrl -v"
    echo "  2. Terminal 2: ./test_preempt_ctrl 0  (runs forever)"
    echo "  3. Terminal 1: list                    (see TSGs)"
    echo "  4. Terminal 1: preempt-pid <pid>       (preempt the CUDA process)"
    echo ""
}

# Main
case "${1:-all}" in
    check)
        check_tracepoints
        ;;
    build)
        build_tool
        build_cuda_test
        ;;
    test)
        run_quick_test
        ;;
    monitor)
        echo "Starting gpu_preempt_ctrl in verbose mode..."
        echo "Press Ctrl+C to stop"
        sudo ./gpu_preempt_ctrl -v
        ;;
    cuda)
        if [ -x ./test_preempt_ctrl ]; then
            ./test_preempt_ctrl "${2:-5}" "${3:-1000000000}"
        else
            echo "CUDA test not built. Run: $0 build"
        fi
        ;;
    all)
        check_tracepoints || true
        build_tool
        build_cuda_test || true
        echo -e "${GREEN}Build complete!${NC}"
        echo ""
        echo "To test interactively:"
        echo "  Terminal 1: sudo ./gpu_preempt_ctrl -v"
        echo "  Terminal 2: ./test_preempt_ctrl 0"
        echo ""
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo "Unknown command: $1"
        usage
        exit 1
        ;;
esac
