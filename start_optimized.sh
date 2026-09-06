#!/bin/bash

# Jetson AGX Orin Optimization Script

echo "Setting up optimized environment for DAM4SAM..."

# Set CPU governor to performance
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Set GPU frequency to maximum
sudo nvpmodel -m 0  # MAX-N mode
sudo jetson_clocks

# Disable desktop effects (if running with GUI)
gsettings set org.gnome.desktop.interface enable-animations false

# Set environment variables for PyTorch
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUDA_LAUNCH_BLOCKING=0

# Set PyTorch JIT to optimize
export PYTORCH_JIT=1

echo "Starting optimized tracker..."
python3 main.py
