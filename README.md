# Real-Time Acceleration for Medical Image Processing

A MATLAB-based medical imaging and deep learning project for breast ultrasound image classification using transfer learning, GPU acceleration, and a graphical user interface.

## Project Overview

This project develops an AI-assisted breast ultrasound image classification pipeline using MATLAB.

The system accepts a breast ultrasound image and classifies it into one of three categories:

- **Benign**
- **Malignant**
- **Normal**

The final classification model is a **ResNet-18 network adapted using transfer learning** and trained on the BUSI (Breast Ultrasound Images) dataset.

The project also demonstrates GPU-based deep learning using an NVIDIA RTX 4050 Laptop GPU and provides a MATLAB-based graphical interface for image analysis.

## Project Objectives

The main objectives of the project are:

1. Develop a medical ultrasound image preprocessing pipeline.
2. Build and evaluate a baseline CNN for breast ultrasound classification.
3. Improve classification using transfer learning with ResNet-18.
4. Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
5. Utilize NVIDIA GPU acceleration for deep learning inference.
6. Develop a MATLAB GUI for practical image classification.
7. Explore the architecture required for real-time medical image processing and future edge deployment.

## System Pipeline

```text
Breast Ultrasound Image
          |
          v
Image Preprocessing
  - Image reading
  - Grayscale conversion
  - Resize to 224 x 224
  - Conversion to 3 channels
          |
          v
ResNet-18
Transfer Learning
          |
          v
3-Class Classification
  +-------------------+
  | Benign            |
  | Malignant         |
  | Normal            |
  +-------------------+
          |
          v
Prediction + Confidence
          |
          v
MATLAB GUI
```

## NVIDIA Holoscan C++ Real-Time Inference Pipeline

The repository includes a high-performance C++ real-time inference pipeline built on the **NVIDIA Holoscan SDK (v4.6)** and **TensorRT**.

### Environment Setup (WSL Ubuntu-22.04)

- **OS:** Ubuntu 22.04 LTS (WSL2)
- **SDK:** NVIDIA Holoscan SDK 4.6 (`/opt/nvidia/holoscan`)
- **CUDA:** CUDA Toolkit 13.4 (`/usr/local/cuda-13.4`)
- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)

### Build Instructions

Build the application from the canonical Windows checkout through WSL Bash:

```bash
# Run from WSL Ubuntu-22.04
cd /mnt/c/Users/anura/Desktop/RealTimeMedicalImaging
./scripts/build_holoscan.sh
```

By default, build artifacts are stored in `~/holoscan_build` (outside the Git working tree) to keep the repository clean.

### Execution Instructions

Run the compiled pipeline:

```bash
# Default run (100 measured frames + 10 warmup frames)
./scripts/run_holoscan.sh

# Fast 5-frame smoke test
STREAM_MEASURED=5 ./scripts/run_holoscan.sh
```

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `ULTRASOUND_RF_PATH` | Path to raw RF sample binary file (`rf_data.bin`) for CUDA GPU DAS reconstruction mode | Unset (uses dataset or single image mode) |
| `ULTRASOUND_IMAGE_PATH` | Path to single PNG/JPG image file for single-image replay mode | Unset |
| `ULTRASOUND_ROOT` | Path to BUSI dataset directory containing `benign/`, `malignant/`, `normal/` subdirectories | `$PROJECT_ROOT/data/Dataset_BUSI_with_GT` |
| `STREAM_MEASURED` | Number of frames to benchmark after initial 10 warm-up frames | `100` |
| `HOLOSCAN_MODEL_PATH` | Path to fine-tuned ResNet-18 ONNX model | `$PROJECT_ROOT/holoscan/models/resnet18_busi.onnx` |
| `BUILD_DIR` | Output directory for Holoscan build artifacts and TRT engine | `${HOME}/holoscan_build` |

### Troubleshooting & Common Failure Checks

1. **Dataset directory missing:** Ensure `data/Dataset_BUSI_with_GT` exists in the canonical checkout or set `ULTRASOUND_ROOT` to point to a valid dataset location.
2. **ONNX model missing:** Ensure `holoscan/models/resnet18_busi.onnx` is present or set `HOLOSCAN_MODEL_PATH`.
3. **CUDA Toolkit not found:** Ensure CUDA 13.4 is installed at `/usr/local/cuda-13.4` or `/usr/local/cuda`.