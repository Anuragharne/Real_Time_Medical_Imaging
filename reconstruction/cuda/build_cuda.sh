#!/usr/bin/env bash
# ==============================================================================
# Build CUDA DAS Beamforming Executable with cuFFT in WSL
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Auto-detect CUDA toolkit
if [ -d "/usr/local/cuda-13.4" ]; then
    export CUDA_HOME="/usr/local/cuda-13.4"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

echo "=========================================================="
echo " Building CUDA DAS Reconstruction Prototype (cuFFT Enabled)"
echo " CUDA NVCC: $(which nvcc)"
echo " CUDA Ver : $(nvcc --version | grep release)"
echo "=========================================================="

cd "$SCRIPT_DIR"

nvcc -O3 \
    -arch=native \
    das_kernel.cu main_das_cuda.cpp \
    -lcufft \
    -o main_das_cuda

if [ -x "main_das_cuda" ]; then
    echo "=========================================================="
    echo " Build SUCCESSFUL!"
    echo " Binary: $SCRIPT_DIR/main_das_cuda"
    echo "=========================================================="
else
    echo "Error: CUDA Build failed." >&2
    exit 1
fi
