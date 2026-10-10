# RF-to-Ultrasound DAS Reconstruction Prototype Report

**Date:** October 10, 2026  
**Environment:** Windows 11 / MATLAB R2025b / WSL2 Ubuntu-22.04 LTS  
**GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)  
**Module Directory:** `reconstruction/`  
**Primary Source Files:** `reconstruction/simulateRFData.m`, `reconstruction/reconstructDAS.m`, `reconstruction/validateReconstruction.m`  
**Generated Integration Artifact:** `docs/rf_bmode_preview.png`  

---

## 1. Executive Summary

A complete, physically meaningful raw Radio-Frequency (RF) to 2-D B-mode ultrasound image reconstruction prototype was implemented, spatially validated, and integrated into NVIDIA Holoscan.

- **Acoustic & Array Model:** 64-element linear array transducer, $f_0 = 5.0\text{ MHz}$, $f_s = 40.0\text{ MHz}$, speed of sound $c = 1540\text{ m/s}$, element pitch $d = 0.308\text{ mm}$.
- **Reconstruction Algorithm:** Delay-and-Sum (DAS) plane-wave beamforming, analytic signal envelope extraction (FFT Hilbert transform), $50\text{ dB}$ log-compression, and $224 \times 224 \times 3$ uint8 B-mode image formatting.
- **Validation:** 100% finite (no `NaN`/`Inf`), valid pixel range ($0–255$), and execution latency of **541.366 ms** recorded in MATLAB.
- **2D Spatial Sanity Check:** Sub-millimeter peak positioning accuracy ($< 0.42\text{ mm}$ error across all 4 scatterer targets at depths $15\text{ mm}$, $25\text{ mm}$, $35\text{ mm}$).
- **Holoscan Integration:** Successfully connected via `ULTRASOUND_IMAGE_PATH` environment variable. Pipeline achieved **49.55 FPS (20.18 ms/frame)** TensorRT inference on the reconstructed image stream.

---

## 2. Signal Processing Pipeline & Physics Parameters

```text
Raw RF Channel Signals (2338 x 64)
              │
              ▼
   Delay-and-Sum Beamforming
   (Spatial Grid: 224 x 224)
              │
              ▼
   Envelope Extraction
   (Analytic Signal via Hilbert FFT)
              │
              ▼
   Log Compression (50 dB)
              │
              ▼
   2-D Grayscale B-Mode Image
   (224 x 224 x 3 uint8)
              │
              ▼
   NVIDIA Holoscan / TensorRT Engine
   (ResNet-18 3-Class Classification)
```

---

## 3. Prototype Validation & Performance Metrics

| Metric | Result | Target / Expected | Status |
|---|---|---|---|
| **Channel Signal Dimensions** | $2338 \times 64$ samples | Multi-channel RF array | Validated |
| **Output Image Dimensions** | $224 \times 224 \times 3$ | $224 \times 224 \times 3$ uint8 | **PASSED** |
| **Numerical Integrity** | $0$ NaNs, $0$ Infs | Finite numbers only | **PASSED** |
| **Intensity Range** | Min: $0$, Max: $255$ | $0–255$ uint8 | **PASSED** |
| **2D Spatial Accuracy** | $< 0.42\text{ mm}$ position error | Target scatterers at $15, 25, 35\text{ mm}$ | **PASSED** |
| **Reconstruction Time** | **541.366 ms** | Single MATLAB CPU pass | Measured |
| **Holoscan Inference Time** | **20.1814 ms / frame** | TensorRT FP32 Engine on RTX 4050 | **PASSED (49.55 FPS)** |

---

## 4. Exploratory ResNet-18 Classifier Inference

- **Exploratory Class Prediction:** `Benign`
- **Exploratory Class Probabilities:** Benign: $0.996706$, Malignant: $0.002787$, Normal: $0.000506$

> [!IMPORTANT]
> **Exploratory Out-of-Domain Notice:** ResNet-18 was fine-tuned on clinical BUSI B-mode images. Simulated phantom RF reconstruction is out-of-domain. While the reconstructed image is fully compatible in shape ($224 \times 224 \times 3$ uint8) and intensity range ($0–255$) for downstream classifier consumption, its classification accuracy is exploratory and unvalidated.

---

## 5. Next Acceleration Step & Linux Deployment Roadmap

To transition from the current offline MATLAB CPU reconstruction ($\sim 541\text{ ms}$) to real-time CUDA-accelerated streaming ($> 30\text{ FPS}$) in Linux Holoscan:

1. **Distinction of Current Components:**
   - **Windows MEX (`resnetPredict_mex.mexw64`):** Validated in Windows MATLAB (3.22x speedup), but strictly bound to Windows host PE/DLL API. Cannot run in Linux.
   - **Standalone CUDA/C++ Code Generation:** MATLAB GPU Coder can be configured with `-config:lib` (C++ Static/Dynamic Library) to generate standalone CUDA kernel sources (`.cu`/`.cpp`) without MATLAB MEX desktop dependencies.
   - **MATLAB CPU Reconstruction:** Prototype function `reconstructDAS.m` proves physical correctness.
   - **TensorRT Inference:** Already fully integrated, compiled, and benchmarked inside Linux Holoscan C++ (`49.55 FPS`).
2. **Recommended Next Step:**
   - Implement a custom CUDA Delay-and-Sum (DAS) beamforming operator in C++ for Holoscan, or generate standalone CUDA C++ code using MATLAB GPU Coder (`coder.config('lib')`) and cross-compile with `nvcc`/`g++` in WSL Ubuntu-22.04.
   - Combine the CUDA DAS operator directly with Holoscan's `ops::InferenceOp` for zero-copy GPU tensor streaming.
