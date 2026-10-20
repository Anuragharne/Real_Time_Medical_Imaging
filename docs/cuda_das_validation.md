# CUDA GPU Delay-and-Sum (DAS) Beamforming Validation & Holoscan Integration Report

**Date:** October 10, 2026  
**Environment:** WSL2 Ubuntu-22.04 LTS / NVIDIA CUDA Toolkit 13.4 / NVIDIA Holoscan SDK 4.6  
**GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)  
**Module Directory:** `reconstruction/cuda/`  
**Holoscan App Directory:** `holoscan/app/`  
**Primary Source Files:** `reconstruction/cuda/das_kernel.cu`, `reconstruction/cuda/das_kernel.h`, `holoscan/app/main.cpp`, `reconstruction/exportAndCompareCUDA.m`  
**Generated Output Artifact:** `docs/cuda_rf_bmode_preview.png`  

---

## 1. Executive Summary

The standalone CUDA C++ GPU implementation of raw Radio-Frequency (RF) Delay-and-Sum (DAS) beamforming was fully integrated into the NVIDIA Holoscan SDK application (`resnet18_holoscan`). The pipeline now successfully executes end-to-end processing from raw RF simulation data through GPU DAS beamforming, cuFFT analytic envelope detection, log compression, tensor formatting (`[1, 3, 224, 224]` NCHW float32), and TensorRT ResNet-18 inference.

- **Numerical Parity with MATLAB Reference:** **73.04 dB PSNR** ($0.0032$ gray level MAE, $1.000000$ structural correlation).
- **Integrated Holoscan Pipeline Throughput:** **39.81 FPS** ($25.13\text{ ms}$ total end-to-end latency per frame).
- **Inference Verdict on Reconstructed RF:** **Benign** (Probability: $0.9927$).
- **Multi-Mode Support:** Preserved 100% compatibility with single image mode (`ULTRASOUND_IMAGE_PATH`) and full dataset replay mode (`ULTRASOUND_ROOT`).

---

## 2. Pixel Parity & Parity Discrepancy Characterization

To validate numerical parity between MATLAB CPU and CUDA GPU reconstructions, the deterministic raw RF sample (`rf_data.bin`) was processed through both pipelines:

| Metric / Property | MATLAB CPU Reference | CUDA GPU Implementation | Comparison / Metric Value | Status |
|---|---|---|---|---|
| **Linear Envelope Signal PSNR** | Baseline | Baseline | **112.26 dB** ($4.13 \times 10^{-5}$ MAE) | **PASSED** |
| **Display B-Mode Image MAE** | $0 - 255$ gray levels | $0 - 255$ gray levels | **0.0032 gray levels** (< 0.001% error) | **PASSED** |
| **Display B-Mode Image PSNR** | Baseline | Baseline | **73.04 dB** | **PASSED** |
| **Normalized Cross-Correlation (NCC)** | $1.000000$ | $1.000000$ | **1.000000** | **PASSED** |
| **Finite Number Integrity** | 100% Pass ($0$ NaNs, $0$ Infs) | 100% Pass ($0$ NaNs, $0$ Infs) | **Identical** | **PASSED** |

### Investigation & Resolution of Initial Discrepancy:
- **Initial Observation:** An initial validation report recorded a B-mode display MAE of $35.80$ gray levels and PSNR of $14.78\text{ dB}$.
- **Root Cause Identified:** The CUDA C++ kernel (`dasBeamformKernel`) writes 2D spatial grid pixels in C-style **row-major order** (`z_idx * Nx + x_idx`), whereas MATLAB's `fread` + `reshape(..., [224, 224, 3])` interpreted the binary data as column-major without transposing matrix dimensions. This resulted in an un-transposed matrix comparison ($90^\circ$ spatial orientation mismatch).
- **Correction Implemented:** When `exportAndCompareCUDA.m` was updated to properly transpose the C++ row-major matrix (`reshape(..., [Nx, Nz])'`), the B-mode MAE dropped from $35.80$ down to **0.0032 gray levels**, and PSNR rose from $14.78\text{ dB}$ to **73.04 dB**. Floating-point precision differences between MATLAB double-precision and CUDA single-precision account for the tiny $0.0032$ residual difference.

---

## 3. 2D Spatial Target Localization (5 Scatterer Targets)

All 5 simulated acoustic point-scatterer targets are localized by the CUDA beamformer. In the canonical validation script ([`reconstruction/exportAndCompareCUDA.m`](file:///c:/Users/anura/Desktop/RealTimeMedicalImaging/reconstruction/exportAndCompareCUDA.m)), a localized peak window ($\pm 10$ pixels) around each expected target coordinates yields spatial errors of **$0.32\text{--}0.41\text{ mm}$**:

| Target Scatterer | Simulated Target $(x, z)$ | Canonical CUDA Peak $(x, z)$ | Canonical Spatial Error | Historical Unconstrained Search Error |
|---|---|---|---|---|
| **Target 1 (Center, 15 mm)** | $(x=0.0\text{ mm}, z=15.0\text{ mm})$ | $(x=-0.0\text{ mm}, z=15.4\text{ mm})$ | **$0.41\text{ mm}$** | $0.86\text{ mm}$ |
| **Target 2 (Left, 25 mm)** | $(x=-5.0\text{ mm}, z=25.0\text{ mm})$ | $(x=-5.3\text{ mm}, z=25.3\text{ mm})$ | **$0.38\text{ mm}$** | $0.36\text{ mm}$ |
| **Target 3 (Right, 25 mm)** | $(x=+5.0\text{ mm}, z=25.0\text{ mm})$ | $(x=+5.3\text{ mm}, z=25.3\text{ mm})$ | **$0.38\text{ mm}$** | $0.09\text{ mm}$ |
| **Target 4 (Center, 35 mm)** | $(x=0.0\text{ mm}, z=35.0\text{ mm})$ | $(x=0.0\text{ mm}, z=35.3\text{ mm})$ | **$0.32\text{ mm}$** | $0.86\text{ mm}$ |
| **Target 5 (Left, 35 mm)** | $(x=-8.0\text{ mm}, z=35.0\text{ mm})$ | $(x=-8.0\text{ mm}, z=35.3\text{ mm})$ | **$0.32\text{ mm}$** | $1.39\text{ mm}$ |

*Note on Historical Discrepancy:* In an earlier development iteration prior to `exportAndCompareCUDA.m` adopting local windowing around scatterer coordinates, global peak detection produced errors between $0.09\text{ mm}$ and $1.39\text{ mm}$ due to acoustic sidelobe energy interference at deeper depths ($z=35\text{ mm}$). Both sets of figures are retained for historical auditability.

---

## 4. Performance & Latency Boundary Accounting

To provide rigorous system performance accounting, latency is broken down across distinct, non-interchangeable boundaries:

| Processing Stage / Scope | Operations & Data Flow Included | Latency per Frame | Throughput | Ratio vs CPU Baseline |
|---|---|---|---|---|
| **Baseline MATLAB CPU** | Single-threaded CPU beamforming + Hilbert envelope + log compress | $537.48\text{ ms}$ | $1.86\text{ FPS}$ | $1.00\times$ (Baseline) |
| **CUDA GPU Steady-State Kernels** | `dasBeamformKernel` + cuFFT C2C + Hilbert filter + Envelope + LogCompress | **0.133 ms** | **7,514.86 FPS** | **$4,041\times$** (Kernel level) |
| **CUDA GPU + PCIe Data Transfers** | Includes Host-to-Device RF transfer & Device-to-Host B-mode transfer | **1.175 ms** | **851.06 FPS** | **$457\times$** (GPU module) |
| **Holoscan Integrated RF Pipeline** | Full Holoscan app: RF load + CUDA DAS + NCHW Float32 formatting + TRT ResNet-18 | **25.13 ms** | **39.81 FPS** | **$21.4\times$** (System level) |

> [!IMPORTANT]
> **Boundary Accounting Rules:**
> - The **0.133 ms** timing represents pure steady-state GPU kernel execution time (excluding PCIe transfers, file I/O, and host tensor formatting).
> - The **1.175 ms** timing includes PCIe host-to-device and device-to-host memory copy overheads.
> - The **25.13 ms** timing represents the complete end-to-end streaming latency of the Holoscan C++ application, including single-iteration CUDA DAS execution, Host memory format conversion into `[1, 3, 224, 224]` float32 NCHW tensor, GXF message routing, and TensorRT FP32 ResNet-18 model execution on the RTX 4050 GPU.

---

## 5. Holoscan Integration Verification

The Holoscan application (`resnet18_holoscan`) was tested across all 3 supported input modes:

```
========================================
   End-to-End Holoscan Stream Result
========================================
Stream Mode           : CUDA RF Beamforming
Input Source          : rf_data.bin
Measured frames       : 5
Total measured time   : 0.125633 s
Effective time/frame  : 25.13 ms
Throughput            : 39.81 frames/s

Final frame prediction
Benign    : 0.992676
Malignant : 0.007137
Normal    : 0.000189
Prediction : Benign
========================================
```

| Mode | Input Source | Latency / Frame | Throughput | ResNet-18 Prediction |
|---|---|---|---|---|
| **CUDA RF DAS Reconstruct** | `rf_data.bin` (Simulated Raw RF) | **25.13 ms** | **39.81 FPS** | **Benign** ($0.9927$) |
| **Single Image Replay** | `docs/rf_bmode_preview.png` | **22.35 ms** | **44.73 FPS** | **Benign** ($0.9967$) |
| **BUSI Dataset Replay** | `data/Dataset_BUSI_with_GT` | **94.52 ms** | **10.58 FPS** | **Normal** ($0.9999$) |

---

## 6. Project Milestone Disclaimers

> [!NOTE]
> **Clinical & Hardware Disclaimer:** This implementation uses simulated offline RF replay data (`rf_data.bin`) from a 5-scatterer physics phantom model. It demonstrates software/hardware integration of GPU-accelerated beamforming within NVIDIA Holoscan, but does not represent live hardware ADC acquisition or clinical diagnostic validation.
