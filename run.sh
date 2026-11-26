
#!/bin/bash

set -e

echo "======================================"
echo " CUDA SIGNAL PROCESSING CAPSTONE"
echo "======================================"

# Create required directories
mkdir -p data output results

# Compile signal generator
echo "[1/5] Compiling signal generator..."

g++ -O2 -std=c++17 signal_generator.cpp -o signal_generator

# Compile CPU processing
echo "[2/5] Compiling CPU processing..."

g++ -O2 -std=c++17 cpu_processing.cpp -lfftw3 -o cpu_processing

# Compile GPU processing
echo "[3/5] Compiling GPU processing..."

nvcc -O2 -std=c++17 gpu_processing.cu -lcufft -o gpu_processing

# Generate datasets
echo "[4/5] Generating input signals..."

./signal_generator 256 4096 2048 0.2 data/input_signals.csv

./signal_generator 1024 4096 2048 0.2 data/input_1024.csv

# Execute processing
echo "[5/5] Running CPU and GPU processing..."

./cpu_processing 256 4096 250 data/input_signals.csv output/cpu_output.csv

./gpu_processing 256 4096 250 data/input_signals.csv output/gpu_output.csv

./cpu_processing 1024 4096 250 data/input_1024.csv output/cpu_1024.csv

./gpu_processing 1024 4096 250 data/input_1024.csv output/gpu_1024.csv

echo "======================================"
echo " PROJECT EXECUTION COMPLETED!"
echo "======================================"

