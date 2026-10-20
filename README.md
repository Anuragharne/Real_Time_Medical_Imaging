# Real-Time Acceleration for Medical Image Processing

A hybrid MATLAB R2025b and C++ NVIDIA Holoscan system for breast ultrasound image processing, physics-based Radio-Frequency (RF) Delay-and-Sum (DAS) beamforming, transfer learning classification, and TensorRT GPU acceleration.

---

## 1. Executive Summary & Project Achievements

This repository implements an end-to-end research and demonstration system for medical ultrasound image processing and beamforming acceleration. It combines **MATLAB R2025b** (Windows) for algorithm prototyping, RF simulation, and model fine-tuning with **NVIDIA Holoscan SDK (v4.6)** (Linux / WSL2) powered by **CUDA 13.4** and **TensorRT** for high-throughput streaming deployment on an NVIDIA GeForce RTX 4050 Laptop GPU.

### Key System Highlights:
- **Transfer Learning Classifier:** ResNet-18 fine-tuned on the Breast Ultrasound Images (BUSI) dataset (780 clinical images across **Benign**, **Malignant**, and **Normal** classes). Historically reported test accuracy on the held-out split is **80.17%** (Benign F1: 83.47%, Malignant F1: 81.97%, Normal F1: 68.18%).
- **Native GPU DAS Beamforming:** Custom CUDA C++ Delay-and-Sum (DAS) beamformer ([`das_kernel.cu`](file:///c:/Users/anura/Desktop/RealTimeMedicalImaging/reconstruction/cuda/das_kernel.cu)) executing parallel delay calculation, sub-sample parabolic interpolation, batched 1D cuFFT Hilbert transform for analytic envelope detection, and logarithmic compression in **0.133 ms kernel execution time** ($1.175\text{ ms}$ including PCIe memory transfers).
- **Parity Verified against MATLAB CPU:** Achieved **73.04 dB B-mode PSNR** ($0.0032$ gray-level MAE on a $0\text{--}255$ scale, $1.000000$ Normalized Cross-Correlation) and **112.26 dB linear-envelope PSNR** between MATLAB CPU double-precision reference code and CUDA single-precision beamforming.
- **NVIDIA Holoscan Streaming Application:** Modular C++ application ([`main.cpp`](file:///c:/Users/anura/Desktop/RealTimeMedicalImaging/holoscan/app/main.cpp)) supporting live simulated RF replay, single-image replay, and dataset streaming directly into a TensorRT FP32 execution engine.
- **Interactive MATLAB Desktop GUI:** Portable desktop application ([`ultrasoundApp.m`](file:///c:/Users/anura/Desktop/RealTimeMedicalImaging/inference/ultrasoundApp.m)) featuring dynamic path resolution and single-click diagnostic inference.
- **Architecture & Demonstration Guides:** Complete vector architecture diagram ([`pipeline_architecture.svg`](file:///c:/Users/anura/Desktop/RealTimeMedicalImaging/docs/pipeline_architecture.svg)), operator dataflow specifications ([`pipeline_architecture.md`](file:///c:/Users/anura/Desktop/RealTimeMedicalImaging/docs/pipeline_architecture.md)), and reproducible demonstration run guide with shot list ([`demo_run_guide.md`](file:///c:/Users/anura/Desktop/RealTimeMedicalImaging/docs/demo_run_guide.md)).

---

## 2. System Architecture & Topology

The system separates offline development and desktop simulation from Linux production streaming:

```text
+---------------------------------------------------------------------------------------+
|                                    DATA SOURCES                                       |
|  • BUSI Clinical B-Mode Dataset (780 PNGs)  • Prerecorded Simulated RF (rf_data.bin)  |
+---------------------------------------------------------------------------------------+
                                           |
         +---------------------------------+---------------------------------+
         v                                                                   v
+------------------------------------+             +------------------------------------+
|  Windows 11 / MATLAB R2025b        |             |  Linux WSL2 Ubuntu 22.04 LTS       |
|  (Prototyping & Desktop GUI)       |             |  (Production Streaming Pipeline)   |
+------------------------------------+             +------------------------------------+
| • ResNet-18 DAGNetwork Fine-Tuning |             | • NVIDIA Holoscan SDK 4.6 (C++)    |
| • Physics Acoustic RF Simulation   |             | • Native CUDA DAS (das_kernel.cu)  |
| • Reference CPU DAS (541 ms)       |             | • cuFFT Hilbert Envelope Detection |
| • GPU Coder Windows MEX (2.57 ms)  |             | • CUDA-Pinned Host GXF Tensors     |
| • Desktop GUI (ultrasoundApp.m)    |             | • TensorRT FP32 Engine Inference   |
| • Parity Script (exportAndCompare) |             | • Up to 246 FPS Streaming Rate     |
+------------------------------------+             +------------------------------------+
```

Detailed visual diagrams and operator specifications:
- [Pipeline Architecture Diagram (SVG)](file:///c:/Users/anura/Desktop/RealTimeMedicalImaging/docs/pipeline_architecture.svg)
- [Operator Flow Documentation (Markdown & Mermaid)](file:///c:/Users/anura/Desktop/RealTimeMedicalImaging/docs/pipeline_architecture.md)

---

## 3. Repository Structure

```text
RealTimeMedicalImaging/
├── acceleration/                   # MATLAB GPU acceleration & Coder artifacts
│   ├── gpuSimpleTest.m             # Test function for GPU Coder validation
│   ├── gpuSimpleTest_mex.mexw64    # Compiled Windows MEX test binary
│   ├── resnetPredict.m             # GPU Coder entry point for ResNet-18
│   └── resnetPredict_mex.mexw64    # Validated Windows GPU Coder MEX (3.22x speedup)
├── data/                           # Evaluation & validation data
│   └── Dataset_BUSI_with_GT/       # Breast Ultrasound Images dataset (benign/malignant/normal)
├── docs/                           # Technical reports, architecture diagrams & run guides
│   ├── baseline_benchmark.md       # Initial classifier benchmarks & environment audit
│   ├── checkpoint_b_enhancement_gpu_coder.md # Wiener filter evaluation & Linux GPU Coder audit
│   ├── cuda_das_validation.md      # CUDA DAS kernel benchmarking & profiling report
│   ├── demo_run_guide.md           # Reproducible demonstration run guide & recording shot list
│   ├── gpu_coder_validation.md     # Windows GPU Coder MEX validation & Linux analysis
│   ├── image_enhancement_evaluation.md # CLAHE & Wiener filter evaluation report
│   ├── pipeline_architecture.md    # System architecture & Holoscan operator specifications
│   ├── pipeline_architecture.svg   # Vector architecture & dataflow diagram
│   ├── rf_bmode_preview.png        # Reference reconstructed B-mode preview image
│   └── rf_reconstruction.md        # Physics RF simulation & CPU DAS reconstruction report
├── holoscan/                       # NVIDIA Holoscan deployment tree
│   ├── app/
│   │   ├── CMakeLists.txt          # CMake configuration for Holoscan C++ application
│   │   ├── main.cpp                # Streaming application & GXF operators
│   │   └── resnet18.yaml           # Operator inference parameter configuration
│   └── models/
│       └── resnet18_busi.onnx      # ONNX export of fine-tuned ResNet-18 model
├── inference/                      # MATLAB inference & GUI
│   ├── predictUltrasound.m         # Portable CLI inference script
│   └── ultrasoundApp.m             # Standalone interactive MATLAB desktop GUI
├── models/                         # Trained model binaries
│   ├── createCNN.m                 # 3-layer baseline CNN architecture definition
│   └── netResNet.mat               # Fine-tuned ResNet-18 DAGNetwork
├── preprocessing/                  # Preprocessing utilities
│   ├── augmentData.m               # Data augmentation pipeline
│   └── preprocessImage.m           # Standard grayscale resize & channel replicate
├── reconstruction/                 # RF simulation & DAS beamforming
│   ├── exportAndCompareCUDA.m      # Canonical MATLAB vs. CUDA numerical parity validation
│   ├── reconstructDAS.m            # Reference MATLAB CPU Delay-and-Sum beamforming
│   ├── simulateRFData.m            # 5-scatterer acoustic phantom RF signal simulator
│   ├── validateReconstruction.m    # Reconstruction validation test script
│   └── cuda/                       # Native CUDA C++ beamformer source
│       ├── build_cuda.sh           # Standalone CUDA build script
│       ├── das_kernel.cu           # GPU DAS beamformer & cuFFT Hilbert envelope kernel
│       ├── das_kernel.h            # C++ API header for CUDA DAS module
│       ├── main_das_cuda.cpp       # Standalone CUDA benchmark runner
│       ├── params.bin              # Acquisition metadata binary
│       └── rf_data.bin             # Simulated RF signal binary (598.5 KB)
└── scripts/                        # Automated build & execution scripts
    ├── build_holoscan.sh           # Builds Holoscan binary from canonical source
    └── run_holoscan.sh             # Launches Holoscan application in any input mode
```

---

## 4. Environment & Prerequisites

### Host Environment (Windows 11):
- **MATLAB:** R2025b with Deep Learning Toolbox, Image Processing Toolbox, Parallel Computing Toolbox, and MATLAB Coder / GPU Coder.
- **C++ Compiler:** Microsoft Visual C++ 2022 (MSVC 17.0).
- **CUDA Toolkit:** v12.1.
- **Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM, Compute Capability 8.9).

### Linux Deployment Environment (WSL2 Ubuntu 22.04 LTS):
- **NVIDIA Holoscan SDK:** v4.6.0 (`/opt/nvidia/holoscan`).
- **CUDA Toolkit:** v13.4 (`/usr/local/cuda-13.4` with `nvcc`).
- **NVIDIA TensorRT:** v10.16.1.
- **OpenCV:** v4.5.4 (`libopencv-dev`).
- **CMake:** v3.20+ (v3.31.12 installed).
- **C++ Compiler:** GCC / G++ 11.4.0.

---

## 5. Build & Execution Instructions

### 1. Build the Holoscan C++ Application (Linux / WSL2)

Compiles [`holoscan/app/main.cpp`](file:///c:/Users/anura/Desktop/RealTimeMedicalImaging/holoscan/app/main.cpp) and links [`reconstruction/cuda/das_kernel.cu`](file:///c:/Users/anura/Desktop/RealTimeMedicalImaging/reconstruction/cuda/das_kernel.cu) with Holoscan, TensorRT, OpenCV, and cuFFT. Output binary is placed in `~/holoscan_build/resnet18_holoscan`:

```bash
cd /mnt/c/Users/anura/Desktop/RealTimeMedicalImaging
./scripts/build_holoscan.sh
```

### 2. Run Holoscan Streaming Demonstrations

#### Mode A: Prerecorded Simulated RF Replay (CUDA DAS + TensorRT)
Streams raw simulated RF signals through native GPU DAS beamforming, cuFFT envelope extraction, dynamic range mapping, and TensorRT inference:

```bash
ULTRASOUND_RF_PATH="/mnt/c/Users/anura/Desktop/RealTimeMedicalImaging/reconstruction/cuda/rf_data.bin" \
STREAM_MEASURED=50 \
./scripts/run_holoscan.sh
```

*Verified Output (October 20, 2026):*
- **Effective Latency:** $4.05\text{ ms/frame}$
- **Throughput:** $246.69\text{ FPS}$ (50 measured frames, 10 warm-up)
- **Scores:** Benign: $0.1608$, Malignant: $0.8256$, Normal: $0.0136$ (Prediction: Malignant)

#### Mode B: Single-Image B-Mode Replay
Replays a reconstructed B-mode ultrasound image through OpenCV preprocessing and TensorRT inference:

```bash
ULTRASOUND_IMAGE_PATH="/mnt/c/Users/anura/Desktop/RealTimeMedicalImaging/docs/rf_bmode_preview.png" \
STREAM_MEASURED=50 \
./scripts/run_holoscan.sh
```

*Verified Output (October 20, 2026):*
- **Effective Latency:** $31.53\text{ ms/frame}$ (dominated by per-frame disk I/O via `cv::imread`)
- **Throughput:** $31.71\text{ FPS}$ (50 measured frames, 10 warm-up)
- **Scores:** Benign: $0.9967$, Malignant: $0.0028$, Normal: $0.0005$ (Prediction: Benign, matching MATLAB within $7 \times 10^{-5}$)

#### Mode C: BUSI Clinical Dataset Replay
Streams sequential images from the local BUSI dataset:

```bash
ULTRASOUND_ROOT="/mnt/c/Users/anura/Desktop/RealTimeMedicalImaging/data/Dataset_BUSI_with_GT" \
STREAM_MEASURED=50 \
./scripts/run_holoscan.sh
```

---

## 6. MATLAB / CUDA Numerical Parity Validation

To verify the mathematical equivalence of the native CUDA single-precision beamformer against double-precision MATLAB reference code:

```matlab
% In Windows MATLAB R2025b:
cd('c:\Users\anura\Desktop\RealTimeMedicalImaging\reconstruction');
exportAndCompareCUDA;
```

### Parity Metrics Summary:
- **Linear Envelope PSNR:** **112.26 dB** (Envelope MAE: $4.135 \times 10^{-5}$)
- **B-Mode Image PSNR:** **73.04 dB** (B-Mode MAE: **0.0032 gray levels** on a $0\text{--}255$ scale)
- **Normalized Cross-Correlation (NCC):** **1.000000**
- **Target Spatial Accuracy:** In canonical `exportAndCompareCUDA.m`, scatterer localization error across all 5 targets is **0.32–0.41 mm** (Target 1: $0.41\text{ mm}$, Target 2: $0.38\text{ mm}$, Target 3: $0.38\text{ mm}$, Target 4: $0.32\text{ mm}$, Target 5: $0.32\text{ mm}$). In earlier unconstrained peak search without local windowing, errors ranged from $0.09\text{--}1.39\text{ mm}$.

---

## 7. Performance Accounting & Measurement Conditions

To maintain scientific integrity, latency figures must be interpreted within their explicit measurement conditions:

| Processing Stage | Implementation / Scope | Latency per Frame | Effective Throughput | Measurement Conditions |
|---|---|---|---|---|
| **MATLAB CPU DAS** | `reconstructDAS.m` (CPU double-precision) | $537\text{--}541\text{ ms}$ | $1.86\text{ FPS}$ | Single-threaded Intel i7 host CPU |
| **CUDA DAS Pure Kernel** | `dasBeamformKernel` + cuFFT C2C + LogCompress | **0.133 ms** | **7,515 FPS** | Steady-state GPU kernel execution time (RTX 4050) |
| **CUDA DAS Module** | Kernel + Host/Device PCIe Data Transfers | **1.175 ms** | **851 FPS** | Standalone C++ test harness with memory copies |
| **TensorRT FP32 Inference** | `resnet18_busi.engine` (FP32) | **1.56 ms** | **641 FPS** | TensorRT standalone context execution |
| **Holoscan RF Replay (50-Frame Run)** | End-to-end Holoscan RF replay (`task-1258`) | **4.05 ms** | **246.7 FPS** | 50 measured frames (0.2027 s total), 10 warm-up frames, in-memory RF binary |
| **Holoscan RF Replay (5-Frame Run)** | End-to-end Holoscan RF replay (`task-1016`) | **8.73 ms** | **114.6 FPS** | 5 measured frames (0.0436 s total), 10 warm-up frames, in-memory RF binary |
| **Holoscan RF Replay (Historical Run)** | Historical recorded run (`docs/cuda_das_validation.md`) | **25.13 ms** | **39.8 FPS** | 5 measured frames (0.1256 s total), 10 warm-up frames |
| **Holoscan Single-Image Replay** | End-to-end Holoscan image replay (`task-1265`) | **31.53 ms** | **31.7 FPS** | 50 measured frames (1.5766 s total), 10 warm-up frames, includes per-frame disk file reads via `cv::imread` |

> [!IMPORTANT]
> **Measurement Accounting Notice:**
> - In RF replay mode, RF data is loaded into host memory once at startup; subsequent iterations execute GPU DAS and TensorRT directly. Latency measured over 50 frames was $4.05\text{ ms/frame}$ ($0.2027\text{ s}$ total), compared to $8.73\text{ ms/frame}$ ($0.0436\text{ s}$ total) in a separate 5-frame run and $25.13\text{ ms/frame}$ ($0.1256\text{ s}$ total) in an earlier recorded 5-frame run. Each figure reflects its exact recorded test invocation and frame count.
> - These figures represent software file replay throughput under specific host/WSL2 conditions and must not be presented as live ultrasound hardware acquisition performance.

---

## 8. Technical Evaluations & Engineering Decisions

### 1. Classifier Performance Accounting
- **Historically Reported Metric:** The fine-tuned ResNet-18 DAGNetwork achieves **80.17% test accuracy** on the held-out 116-image BUSI test split (65 benign, 31 malignant, 20 normal).
- **Per-Class Breakdown:** Benign F1: $83.47\%$, Malignant F1: $81.97\%$, Normal F1: $68.18\%$.
- **Truth in Reporting:** Training and evaluation were executed offline prior to repository handoff; the model binary is saved as [`models/netResNet.mat`](file:///c:/Users/anura/Desktop/RealTimeMedicalImaging/models/netResNet.mat). Unsubstantiated claims of 98.08% accuracy or AUC 1.0 have been removed.

### 2. Image Enhancement Evaluation (CLAHE & Wiener Filtering Rejected)
- **CLAHE Evaluation ([`docs/image_enhancement_evaluation.md`](file:///c:/Users/anura/Desktop/RealTimeMedicalImaging/docs/image_enhancement_evaluation.md)):** Tested across clip limits $0.005\text{--}0.050$. Rejected because contrast stretching destroyed acoustic attenuation shadows, causing a catastrophic false negative on `malignant (1).png` (flipping the diagnosis to Benign with $63.97\%$ confidence).
- **Wiener Filter Evaluation ([`docs/checkpoint_b_enhancement_gpu_coder.md`](file:///c:/Users/anura/Desktop/RealTimeMedicalImaging/docs/checkpoint_b_enhancement_gpu_coder.md)):** Tested with $[3 \times 3]$ and $[5 \times 5]$ adaptive smoothing windows. While diagnostic labels were preserved, confidence on the malignant lesion dropped by $5.60\%$ due to lesion boundary blurring.
- **Decision:** Both enhancement methods were **formally rejected**. The unenhanced native B-mode baseline is retained across all active pipelines.

### 3. GPU Coder Feasibility vs. Native CUDA Holoscan Architecture
- **Windows Desktop Acceleration:** Validated Windows GPU Coder MEX ([`resnetPredict_mex.mexw64`](file:///c:/Users/anura/Desktop/RealTimeMedicalImaging/acceleration/resnetPredict_mex.mexw64)), achieving a **3.22x speedup** ($2.57\text{ ms/frame}$) with numerical error $< 6.0 \times 10^{-7}$ vs. MATLAB double-precision `predict()`. Retained for interactive MATLAB desktop use.
- **Linux GPU Coder Feasibility:** Demonstrated that MATLAB GPU Coder can export standalone C++/CUDA source files (`GenCodeOnly = true`) decoupled from MEX APIs, and compiled them under Linux WSL2 with CUDA 13.4 into a static library ([`libgpusimpletest.a`](file:///c:/Users/anura/Desktop/RealTimeMedicalImaging/docs/checkpoint_b_enhancement_gpu_coder.md#33-standalone-linux-code-generation--compilation-proof-of-concept), 25.8 KB).
- **Deployment Decision:** The native handwritten CUDA DAS kernel ([`das_kernel.cu`](file:///c:/Users/anura/Desktop/RealTimeMedicalImaging/reconstruction/cuda/das_kernel.cu)) and TensorRT FP32 engine are retained for Linux Holoscan deployment. Forcing GPU Coder-generated C++ code into Holoscan GXF operators would introduce MATLAB runtime header dependencies without providing latency or quality benefits.

---

## 9. Disclaimers & Limitations

> [!WARNING]
> **Research & Demonstration Boundaries:**
> 1. **Prerecorded / Simulated RF Data:** The RF dataset (`rf_data.bin`) is simulated from a 5-scatterer acoustic phantom model. The system demonstrates software/hardware pipeline integration within NVIDIA Holoscan and does **not** interface with a physical ultrasound transducer or ADC digitizer.
> 2. **Out-of-Domain Inference:** The ResNet-18 classifier was fine-tuned on clinical B-mode breast ultrasound images. Outputs on synthetic point-scatterer phantom reconstructions reflect algorithmic responses to synthetic acoustic reflections and must not be interpreted as clinical diagnoses.
> 3. **Non-Clinical Software:** This project is an academic research and engineering demonstration of GPU acceleration and is not certified or intended for clinical diagnostic use.