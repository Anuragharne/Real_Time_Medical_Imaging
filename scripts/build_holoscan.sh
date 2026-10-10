#!/usr/bin/env bash
# ==============================================================================
# Build Holoscan ResNet-18 Application from Canonical Checkout
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SOURCE_DIR="$PROJECT_ROOT/holoscan/app"
BUILD_DIR="${BUILD_DIR:-${HOME}/holoscan_build}"

# Auto-detect local CMake if installed in ~/.local
for cmake_bin in "$HOME"/.local/cmake-*/bin; do
    if [ -d "$cmake_bin" ]; then
        export PATH="$cmake_bin:$PATH"
    fi
done

# Auto-detect CUDA toolkit if not already in PATH
if [ -d "/usr/local/cuda-13.4" ]; then
    export CUDA_HOME="/usr/local/cuda-13.4"
    export CUDAToolkit_ROOT="/usr/local/cuda-13.4"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
    export CUDAToolkit_ROOT="/usr/local/cuda"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

echo "=========================================================="
echo " Building Holoscan Application"
echo " Project Root: $PROJECT_ROOT"
echo " Source Dir:   $SOURCE_DIR"
echo " Build Dir:    $BUILD_DIR"
echo " CMake Path:   $(which cmake)"
echo " CMake Ver:    $(cmake --version | head -n 1)"
echo " CUDA Home:    ${CUDA_HOME:-Not Set}"
echo "=========================================================="

if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' does not exist." >&2
    exit 1
fi

if [ ! -f "$SOURCE_DIR/CMakeLists.txt" ]; then
    echo "Error: CMakeLists.txt not found in '$SOURCE_DIR'." >&2
    exit 1
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
if [ -f "CMakeCache.txt" ]; then
    rm -f CMakeCache.txt
fi

echo "[1/2] Running CMake configuration..."
cmake "$SOURCE_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    ${CUDAToolkit_ROOT:+-DCUDAToolkit_ROOT="$CUDAToolkit_ROOT"}

echo "[2/2] Building binary..."
cmake --build . -j"$(nproc)"

BINARY="$BUILD_DIR/resnet18_holoscan"
if [ -x "$BINARY" ]; then
    echo "=========================================================="
    echo " Build SUCCESSFUL!"
    echo " Executable: $BINARY"
    echo "=========================================================="
else
    echo "Error: Build finished but executable '$BINARY' was not created." >&2
    exit 1
fi
