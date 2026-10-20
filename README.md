# Real-Time Acceleration for Medical Image Processing

A hybrid MATLAB / C++ NVIDIA Holoscan system for breast ultrasound image processing, physics-based Radio-Frequency (RF) Delay-and-Sum (DAS) beamforming, transfer learning classification, and TensorRT GPU acceleration.

---

## 1. Executive Summary & Project Achievements

This project implements an end-to-end AI-assisted breast ultrasound classification and reconstruction system using **MATLAB R2025b** and the **NVIDIA Holoscan SDK (v4.6)** powered by **CUDA 13.4** and **TensorRT**.

### Key System Highlights:
- **Transfer Learning Classifier:** ResNet-18 fine-tuned on the BUSI (Breast Ultrasound Images) dataset, classifying ultrasound scans into 3 categories (**Benign**, **Malignant**, **Normal**). Historically reported test accuracy on the held-out split is **80.17%** (Benign F1: 83.47%, Malignant F1: 81.97%, Normal F1: 68.18%).
- **GPU-Accelerated DAS Beamforming:** Custom CUDA C++ Delay-and-Sum (DAS) beamformer executing parallel delay calculation, fractional sample interpolation, 1D batched cuFFT Hilbert transform, and log compression.
- **Parity Verified against MATLAB CPU:** Achieved **73.04 dB PSNR** ($0.0032$ gray levels MAE, $1.000000$ structural correlation) between MATLAB CPU double-precision reference and CUDA GPU single-precision beamforming.
- **NVIDIA Holoscan C++ Pipeline:** Integrated streaming application supporting live CUDA RF reconstruction, single-image replay, and dataset streaming directly into TensorRT ResNet-18 inference at **39.81 FPS** ($25.13\text{ ms}$ latency).

---

## 2. System Architecture & Component Responsibilities

The system is architected across a hybrid Windows / Linux (WSL2) environment:

```text
                                +-----------------------------------+
                                |      Simulated RF Data / BUSI     |
                                +-----------------------------------+
                                                  |
                     +----------------------------+----------------------------+
                     |                                                         |
                     v                                                         v
        +--------------------------+                              +--------------------------+
        |  MATLAB R2025b (Windows) |                              |  NVIDIA Holoscan (WSL2)  |
        +--------------------------+                              +--------------------------+
        | • Physics RF Signal Sim  |                              | • Custom CUDA C++ DAS    |
        | • CPU DAS Reference      |                              | • cuFFT Envelope Detect  |
        | • Model Fine-Tuning      |                              | • Float32 NCHW Tensor    |
        | • GPU Coder MEX Valid.   |                              | • TensorRT Engine Infer. |
        | • Interactive MATLAB GUI |                              | • 39.81 FPS End-to-End   |
        +--------------------------+                              +--------------------------+
```

### Environment Responsibilities:
- **Windows (MATLAB R2025b):** Used for baseline CNN architecture definition, ResNet-18 transfer learning model host (`models/netResNet.mat`), physics-based RF acoustic phantom simulation (`simulateRFData.m`), reference CPU beamforming (`reconstructDAS.m`), GPU Coder MEX validation, CLAHE enhancement evaluation, and interactive MATLAB GUI (`inference/ultrasoundApp.m`).
- **Linux / WSL2 (Ubuntu 22.04 LTS):** Production deployment environment running C++ NVIDIA Holoscan SDK 4.6, CUDA 13.4, and TensorRT. Executes native CUDA DAS kernels (`das_kernel.cu`) and feeds NCHW float32 tensors (`[1, 3, 224, 224]`) directly into TensorRT inference engine on an NVIDIA GeForce RTX 4050 Laptop GPU.

---

## 3. Supported Pipeline Input Modes

The C++ Holoscan application (`resnet18_holoscan`) supports three distinct streaming input paths via environment variables:

| Input Mode | Environment Variable | Input Source | Latency / Frame | Throughput | ResNet-18 Prediction |
|---|---|---|---|---|---|
| **CUDA RF DAS Reconstruct** | `ULTRASOUND_RF_PATH` | `reconstruction/cuda/rf_data.bin` | **25.13 ms** | **39.81 FPS** | **Benign** ($0.9927$) |
| **Single Image Replay** | `ULTRASOUND_IMAGE_PATH` | `docs/rf_bmode_preview.png` | **22.35 ms** | **44.73 FPS** | **Benign** ($0.9967$) |
| **BUSI Dataset Replay** | `ULTRASOUND_ROOT` | `data/Dataset_BUSI_with_GT` | **94.52 ms** | **10.58 FPS** | **Normal** ($0.9999$) |

---

## 4. Environment & Hardware Requirements

### Host Environment:
- **Operating System:** Windows 11 with WSL2 (Ubuntu-22.04 LTS)
- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM, Compute Capability 8.9)
- **Host Software:** MATLAB R2025b with Image Processing & Parallel Computing Toolboxes

### WSL2 Environment:
- **NVIDIA Holoscan SDK:** v4.6.0 (`/opt/nvidia/holoscan`)
- **CUDA Toolkit:** v13.4 (`/usr/local/cuda-13.4`)
- **OpenCV:** v4.5.4
- **CMake:** v3.20+ (v3.31.12 installed)

---

## 5. Build and Execution Instructions (WSL2)

### 1. Build the Holoscan C++ Application

Run the build script from WSL2. Build artifacts are stored in `~/holoscan_build` outside the Git tree:

```bash
cd /mnt/c/Users/anura/Desktop/RealTimeMedicalImaging
./scripts/build_holoscan.sh
```

### 2. Run Pipeline Smoke Tests

#### Mode A: CUDA RF Reconstruction Stream (5 measured frames)
```bash
ULTRASOUND_RF_PATH=/mnt/c/Users/anura/Desktop/RealTimeMedicalImaging/reconstruction/cuda/rf_data.bin STREAM_MEASURED=5 ./scripts/run_holoscan.sh
```

#### Mode B: Single Image Replay Mode
```bash
ULTRASOUND_IMAGE_PATH=/mnt/c/Users/anura/Desktop/RealTimeMedicalImaging/docs/rf_bmode_preview.png STREAM_MEASURED=5 ./scripts/run_holoscan.sh
```

#### Mode C: BUSI Dataset Replay Mode
```bash
STREAM_MEASURED=5 ./scripts/run_holoscan.sh
```

### Environment Variables Reference

| Variable | Description | Default |
|---|---|---|
| `ULTRASOUND_RF_PATH` | Path to raw RF binary file (`rf_data.bin`) for live CUDA DAS reconstruction | Unset |
| `ULTRASOUND_PARAMS_PATH` | Path to DAS acquisition parameters (`params.bin`) | `$(dirname $ULTRASOUND_RF_PATH)/params.bin` |
| `ULTRASOUND_IMAGE_PATH` | Path to single PNG/JPG image file for single-image mode | Unset |
| `ULTRASOUND_ROOT` | Path to BUSI dataset directory | `$PROJECT_ROOT/data/Dataset_BUSI_with_GT` |
| `STREAM_MEASURED` | Number of frames to benchmark after 10 warm-up frames | `100` |
| `HOLOSCAN_MODEL_PATH` | Path to fine-tuned ResNet-18 ONNX model | `$PROJECT_ROOT/holoscan/models/resnet18_busi.onnx` |

---

## 6. Performance Accounting & Measurement Boundaries

To ensure scientific rigor, performance timing is categorized across precise boundaries:

| Scope / Processing Stage | Execution Details | Latency / Frame | Effective Throughput | Speedup vs CPU Baseline |
|---|---|---|---|---|
| **Baseline MATLAB CPU** | Single-threaded CPU beamforming + Hilbert envelope + log compress | $537.48\text{ ms}$ | $1.86\text{ FPS}$ | $1.00\times$ (Baseline) |
| **CUDA GPU Steady-State Kernels** | `dasBeamformKernel` + cuFFT C2C + Hilbert filter + Envelope + LogCompress | **0.133 ms** | **7,514.86 FPS** | **$4,041\times$** (Kernel level) |
| **CUDA GPU + PCIe Data Transfers** | Includes Host-to-Device RF transfer & Device-to-Host B-mode transfer | **1.175 ms** | **851.06 FPS** | **$457\times$** (GPU module) |
| **Integrated Holoscan RF Pipeline** | Full Holoscan app: RF load + CUDA DAS + NCHW Float32 formatting + TRT ResNet-18 | **25.13 ms** | **39.81 FPS** | **$21.4\times$** (System level) |

> [!IMPORTANT]
> **Boundary Accounting Rules:**
> - **0.133 ms** is the pure steady-state GPU kernel execution time.
> - **1.175 ms** includes PCIe host-to-device and device-to-host data transfers.
> - **25.13 ms** represents total end-to-end streaming latency of the Holoscan C++ application, including single-iteration CUDA DAS execution, Host memory format conversion into `[1, 3, 224, 224]` float32 NCHW tensor, GXF message routing, and TensorRT FP32 ResNet-18 execution.

---

## 7. Key Technical Evaluations & Design Decisions

### 1. CLAHE Enhancement Evaluation (Rejected for Active Inference)
- Contrast-Limited Adaptive Histogram Equalization (CLAHE) was evaluated in Checkpoint 4 using clip limits $0.01 - 0.05$.
- **Finding:** While CLAHE improved visual local contrast, it altered ultrasound speckle intensity distributions, degrading ResNet-18 classification performance (e.g. Malignant confidence dropped from $0.9996$ to $0.5739$ or misclassified to Benign).
- **Decision:** CLAHE was **rejected** as an active input transformation. The unenhanced $0 - 255$ image pipeline was retained to preserve classifier accuracy.

### 2. GPU Coder MEX vs Custom CUDA Holoscan Operator
- **GPU Coder MEX (Checkpoint 3):** MATLAB GPU Coder generated MEX code (`resnetPredict_mex.mexw64`) achieved $6.92\text{ ms}$ inference latency with $1.42 \times 10^{-7}$ numerical score difference vs double-precision MATLAB.
- **Custom CUDA C++ Operator (Checkpoint 7-8):** For the RF reconstruction stage, native custom CUDA C++ kernels (`das_kernel.cu`) were written directly in C++/nvcc to avoid MATLAB runtime host dependencies and enable native C++ Holoscan integration.

---

## 8. Summary of Original Brief Objectives vs Project Deliverables

| Original Project Objective | Status | Implementation & Validation Summary |
|---|---|---|
| **1. Medical Image Preprocessing** | **COMPLETED** | Grayscale conversion, $224 \times 224$ resizing, 3-channel replication, NCHW float32 normalization. |
| **2. Baseline CNN Classifier** | **COMPLETED** | 3-layer Custom CNN architecture defined in MATLAB (`models/createCNN.m`). |
| **3. Transfer Learning (ResNet-18)** | **COMPLETED** | Fine-tuned ResNet-18 with historically reported **80.17% test accuracy** on held-out BUSI split (Benign F1: 83.47%, Malignant F1: 81.97%, Normal F1: 68.18%). |
| **4. Comprehensive Model Evaluation** | **COMPLETED** | Evaluated on held-out test split via Precision, Recall, and F1-score per class. (Training/eval was performed offline; model saved as `models/netResNet.mat`). |
| **5. NVIDIA GPU Acceleration (TensorRT)** | **COMPLETED** | Converted ONNX model to TensorRT FP32 engine (`resnet18_busi.engine`), running in Holoscan at **44.73 FPS**. |
| **6. MATLAB GUI Application** | **COMPLETED** | Developed interactive MATLAB GUI (`inference/ultrasoundApp.m`) with live ultrasound image classification. |
| **7. Real-Time Edge Architecture (Holoscan C++)** | **COMPLETED** | Native C++ Holoscan application constructed with custom GXF operators and TensorRT inference. |
| **8. Raw RF Reconstruction (CUDA DAS)** | **COMPLETED (Simulated)** | Custom CUDA C++ DAS beamforming kernel integrated into Holoscan (**39.81 FPS** end-to-end). Verified against MATLAB CPU (**73.04 dB PSNR**). |
| **9. Hardware ADC Live Acquisition** | **NOT COMPLETED / OUT OF SCOPE** | Requires physical ultrasound transducer hardware and PCIe ADC digitizer. Disclosed as offline simulated RF replay. |

---

## 9. Disclaimers & Disclosures

> [!NOTE]
> **Clinical & Hardware Disclaimer:**
> 1. **Offline Simulated RF Data:** The raw RF dataset (`rf_data.bin`) is simulated from a 5-scatterer physics phantom model. It demonstrates software/hardware integration of GPU beamforming within NVIDIA Holoscan, but does not represent live hardware ADC acquisition.
> 2. **Not a Clinical Device:** This software is an engineering demonstration of GPU acceleration for medical image processing and is not certified for clinical diagnostic use.