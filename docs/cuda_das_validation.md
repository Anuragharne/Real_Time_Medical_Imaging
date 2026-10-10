# CUDA GPU Delay-and-Sum (DAS) Beamforming Validation Report

**Date:** October 10, 2026  
**Environment:** WSL2 Ubuntu-22.04 LTS / NVIDIA CUDA Toolkit 13.4  
**GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)  
**Module Directory:** `reconstruction/cuda/`  
**Primary Source Files:** `reconstruction/cuda/das_kernel.cu`, `reconstruction/cuda/das_kernel.h`, `reconstruction/cuda/main_das_cuda.cpp`, `reconstruction/cuda/build_cuda.sh`, `reconstruction/exportAndCompareCUDA.m`  
**Generated Output Artifact:** `docs/cuda_rf_bmode_preview.png`  

---

## 1. Executive Summary

A standalone CUDA C++ GPU implementation of raw Radio-Frequency (RF) Delay-and-Sum (DAS) beamforming was developed, compiled with CUDA 13.4, and validated against the MATLAB CPU reference.

- **GPU Acceleration Architecture:** CUDA C++ kernel (`dasBeamformKernel`) executing 2D thread blocks over the $224 \times 224$ image spatial grid, performing parallel delay calculation, fractional sample interpolation, channel summation, cuFFT analytic envelope detection, and log compression.
- **GPU Latency & Throughput:**
  - **Warm-up Latency:** $2.139\text{ ms}$ (includes initial GPU allocation and CUDA context initialization).
  - **Steady-State Latency:** **0.133 ms per frame** ($7,525.77\text{ FPS}$).
  - **Speedup over Baseline MATLAB CPU:** **$4070\times$ speedup** ($541.37\text{ ms}$ CPU vs $0.133\text{ ms}$ GPU).
- **Spatial Accuracy:** 100% 2D spatial target localization agreement across all 5 simulated scatterer targets with sub-to-near-millimeter position accuracy ($0.09\text{ mm}$ to $1.39\text{ mm}$ error).
- **Deployment Isolation:** Built as a standalone C++ program in `reconstruction/cuda/`. The production C++ Holoscan image-file pipeline remains untouched and 100% functional.

---

## 2. 2D Spatial Scatterer Target Localization (5 Targets)

| Target Scatterer | Simulated Target $(x, z)$ | CUDA GPU Peak $(x, z)$ | Spatial Position Error | Status |
|---|---|---|---|---|
| **Target 1 (Center, 15 mm)** | $(x=0.0\text{ mm}, z=15.0\text{ mm})$ | $(x=+0.1\text{ mm}, z=14.1\text{ mm})$ | $0.86\text{ mm}$ | **PASSED** |
| **Target 2 (Left, 25 mm)** | $(x=-5.0\text{ mm}, z=25.0\text{ mm})$ | $(x=-4.7\text{ mm}, z=24.9\text{ mm})$ | $0.36\text{ mm}$ | **PASSED** |
| **Target 3 (Right, 25 mm)** | $(x=+5.0\text{ mm}, z=25.0\text{ mm})$ | $(x=+5.0\text{ mm}, z=25.1\text{ mm})$ | **$0.09\text{ mm}$** | **PASSED** |
| **Target 4 (Center, 35 mm)** | $(x=0.0\text{ mm}, z=35.0\text{ mm})$ | $(x=+0.1\text{ mm}, z=35.9\text{ mm})$ | $0.86\text{ mm}$ | **PASSED** |
| **Target 5 (Left, 35 mm)** | $(x=-8.0\text{ mm}, z=35.0\text{ mm})$ | $(x=-8.0\text{ mm}, z=36.4\text{ mm})$ | $1.39\text{ mm}$ | **PASSED** |

> [!NOTE]
> **Scatterer Count Clarification:** In Checkpoint 5, 4 target coordinates were highlighted in the brief summary. The simulator definition (`simulateRFData.m`) contains 5 hyperechoic point targets ($15\text{ mm}$, $25\text{ mm}$ left/right, $35\text{ mm}$ center/left). All 5 targets are verified and localized by the CUDA GPU beamformer.

---

## 3. Performance & Latency Comparison

| Implementation | Compute Device | Warm-up Latency | Steady-State Latency per Frame | Throughput | Observed Speedup |
|---|---|---|---|---|---|
| **Baseline MATLAB CPU** | Intel Host CPU | — | 541.366 ms | 1.85 FPS | 1.00x (Baseline) |
| **CUDA C++ DAS Beamformer** | **NVIDIA RTX 4050 Laptop GPU** | **2.139 ms** | **0.133 ms** | **7,525.77 FPS** | **4070x** |

---

## 4. Numerical Parity & Image Integrity

- **Output Dimensions:** $224 \times 224 \times 3$ uint8 (HWC/NCHW compatible).
- **Finite Integrity:** 100% Passed ($0$ NaNs, $0$ Infs).
- **Pixel Value Range:** Min = 0, Max = 255.
- **Preview Artifact:** Saved to `docs/cuda_rf_bmode_preview.png`.

---

## 5. Linux Holoscan Integration Readiness (Checkpoint 8 Roadmap)

- The CUDA kernel (`dasBeamformKernel`) is written in standard CUDA C++ (`nvcc`).
- It can be wrapped cleanly inside a custom Holoscan `Operator` (`ops::CudaDASOp`) in `main.cpp` using CUDA device pointers (`d_bmode`), enabling real-time zero-copy RF array reconstruction directly into Holoscan's `ops::InferenceOp`.
