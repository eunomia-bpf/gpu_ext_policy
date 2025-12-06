#!/bin/bash
# Load custom NVIDIA kernel modules with BPF struct_ops support
# Run with sudo

set -e

DRIVER_DIR="/home/yunwei37/workspace/gpu/open-gpu-kernel-modules/kernel-open"

echo "Loading custom NVIDIA kernel modules..."

# Check if modules exist
if [ ! -f "$DRIVER_DIR/nvidia.ko" ]; then
    echo "Error: nvidia.ko not found in $DRIVER_DIR"
    echo "Please build the kernel modules first"
    exit 1
fi

# Load in order
echo "  Loading nvidia.ko..."
insmod "$DRIVER_DIR/nvidia.ko" NVreg_OpenRmEnableUnsupportedGpus=1 || {
    echo "Failed to load nvidia.ko"
    exit 1
}

echo "  Loading nvidia-modeset.ko..."
insmod "$DRIVER_DIR/nvidia-modeset.ko" || {
    echo "Failed to load nvidia-modeset.ko"
    rmmod nvidia
    exit 1
}

echo "  Loading nvidia-uvm.ko..."
insmod "$DRIVER_DIR/nvidia-uvm.ko" || {
    echo "Failed to load nvidia-uvm.ko"
    rmmod nvidia-modeset nvidia
    exit 1
}

# Optional: load nvidia-drm if needed
if [ -f "$DRIVER_DIR/nvidia-drm.ko" ]; then
    echo "  Loading nvidia-drm.ko..."
    insmod "$DRIVER_DIR/nvidia-drm.ko" modeset=1 || {
        echo "Warning: Failed to load nvidia-drm.ko (optional)"
    }
fi

echo ""
echo "NVIDIA modules loaded successfully!"
echo ""

# Verify BTF is available
if [ -f /sys/kernel/btf/nvidia ]; then
    echo "BTF available: /sys/kernel/btf/nvidia"
else
    echo "Warning: nvidia BTF not found - struct_ops may not work"
fi

# Show loaded modules
echo ""
echo "Loaded modules:"
lsmod | grep nvidia
