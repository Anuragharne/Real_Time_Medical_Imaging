#!/usr/bin/env bash
# ==============================================================================
# Run Holoscan ResNet-18 Application from Canonical Checkout
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BUILD_DIR="${BUILD_DIR:-${HOME}/holoscan_build}"
BINARY="$BUILD_DIR/resnet18_holoscan"

# Auto-detect local CMake if installed in ~/.local
for cmake_bin in "$HOME"/.local/cmake-*/bin; do
    if [ -d "$cmake_bin" ]; then
        export PATH="$cmake_bin:$PATH"
    fi
done

# Auto-detect CUDA environment
if [ -d "/usr/local/cuda-13.4" ]; then
    export CUDA_HOME="/usr/local/cuda-13.4"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

if [ ! -x "$BINARY" ]; then
    echo "Error: Holoscan binary not found at '$BINARY'." >&2
    echo "Please run 'scripts/build_holoscan.sh' first to compile from canonical source." >&2
    exit 1
fi

MODEL_PATH="${HOLOSCAN_MODEL_PATH:-$PROJECT_ROOT/holoscan/models/resnet18_busi.onnx}"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at '$MODEL_PATH'." >&2
    echo "Verify model path or set HOLOSCAN_MODEL_PATH env variable." >&2
    exit 1
fi

MEASURED_FRAMES="${STREAM_MEASURED:-100}"

# Single-image or dataset-replay stream setup
IMAGE_PATH="${ULTRASOUND_IMAGE_PATH:-}"
if [ -n "$IMAGE_PATH" ]; then
    if [ ! -f "$IMAGE_PATH" ]; then
        echo "Error: Specified image file not found at '$IMAGE_PATH'." >&2
        exit 1
    fi
    export ULTRASOUND_IMAGE_PATH="$IMAGE_PATH"
    INPUT_DESC="Single Image: $IMAGE_PATH"
else
    DATASET_ROOT="${ULTRASOUND_ROOT:-$PROJECT_ROOT/data/Dataset_BUSI_with_GT}"
    if [ ! -d "$DATASET_ROOT" ]; then
        echo "Error: Dataset directory not found at '$DATASET_ROOT'." >&2
        echo "Verify dataset path or set ULTRASOUND_ROOT env variable." >&2
        exit 1
    fi
    export ULTRASOUND_ROOT="$DATASET_ROOT"
    INPUT_DESC="Dataset Replay: $DATASET_ROOT"
fi

echo "=========================================================="
echo " Running Holoscan Ultrasound Classifier Pipeline"
echo " Binary:          $BINARY"
echo " Model:           $MODEL_PATH"
echo " Input Source:    $INPUT_DESC"
echo " Measured Frames: $MEASURED_FRAMES (Warm-up: 10)"
echo "=========================================================="

# Create a runtime config with absolute model path to avoid CWD dependencies
RUNTIME_CONFIG="$BUILD_DIR/resnet18_runtime.yaml"
cat << EOF > "$RUNTIME_CONFIG"
inference:
  backend: "trt"

  model_path_map:
    resnet18: "$MODEL_PATH"

  input_map:
    resnet18: ["data"]

  output_map:
    resnet18: ["softmax1000Output"]

  input_on_cuda: false
  output_on_cuda: false
  transmit_on_cuda: false

  enable_fp16: false
  enable_cuda_graphs: false
EOF

export STREAM_MEASURED="$MEASURED_FRAMES"

cd "$BUILD_DIR"
"$BINARY" "$RUNTIME_CONFIG"
