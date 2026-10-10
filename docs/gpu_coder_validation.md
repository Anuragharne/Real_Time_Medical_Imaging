# GPU Coder MEX Validation & Deployment Analysis Report

**Date:** October 10, 2026  
**Environment:** Windows 11 / MATLAB R2025b (Update 7)  
**GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)  
**Target Artifact:** `acceleration/resnetPredict_mex.mexw64`  
**Entry Point Source:** `acceleration/resnetPredict.m`  

---

## 1. Executive Summary

The existing GPU Coder MEX binary (`resnetPredict_mex.mexw64`) was validated against native MATLAB standard inference on the three representative BUSI ultrasound images (`benign (1).png`, `malignant (1).png`, `normal (1).png`).

- **Numerical Precision:** Passed all three samples with maximum absolute score differences below $6.0 \times 10^{-7}$ (far exceeding the $10^{-4}$ target threshold).
- **Classification Parity:** 100% agreement on predicted class labels (`benign`, `malignant`, `normal`).
- **Inference Speedup:** The MEX binary achieved **2.572 ms/frame (388.87 FPS)** compared to standard MATLAB's 8.287 ms/frame (120.67 FPS), delivering a **3.22x speedup** on the RTX 4050 GPU.
- **Deployment Constraint:** The `.mexw64` binary is a Windows-specific MEX DLL dependent on the MATLAB desktop host. It cannot be loaded natively into Linux Holoscan.

---

## 2. Numerical Score Validation (Target $\le 10^{-4}$)

| Sample Image | Class | MATLAB Standard Scores `[Benign, Malignant, Normal]` | GPU Coder MEX Scores `[Benign, Malignant, Normal]` | Max Abs Score Diff | Tolerance Check ($\le 10^{-4}$) | Class Match |
|---|---|---|---|---|---|---|
| `benign (1).png` | Benign | `[0.999983, 0.000015, 0.000002]` | `[0.999983, 0.000015, 0.000002]` | $2.0 \times 10^{-8}$ | **PASSED** | Match (`benign`) |
| `malignant (1).png` | Malignant | `[0.000461, 0.999420, 0.000119]` | `[0.000461, 0.999420, 0.000119]` | $4.8 \times 10^{-7}$ | **PASSED** | Match (`malignant`) |
| `normal (1).png` | Normal | `[0.000161, 0.000001, 0.999839]` | `[0.000161, 0.000001, 0.999838]` | $6.0 \times 10^{-7}$ | **PASSED** | Match (`normal`) |

---

## 3. Performance & Timing Comparison

Timing was measured over 50 repeated inference calls per method following 5 warm-up iterations on an input tensor ($224 \times 224 \times 3$).

| Method | Mean Latency per Frame | Effective Throughput | Relative Speedup |
|---|---|---|---|
| **Standard MATLAB `predict`** | 8.287 ms | 120.67 FPS | 1.00x (Baseline) |
| **GPU Coder MEX (`resnetPredict_mex`)** | **2.572 ms** | **388.87 FPS** | **3.22x** |

---

## 4. Architecture & Linux Holoscan Deployment Limitations

### Existing Artifact Inspection
- **Binary:** `acceleration/resnetPredict_mex.mexw64` (709,632 bytes)
- **Target Platform:** Windows x86_64 (`.mexw64`)
- **Codegen Source Dir:** `acceleration/codegen/mex/resnetPredict/`

### Key Deployment Findings
1. **Windows MEX Dependency:** `.mexw64` is compiled specifically for Windows host environments running MATLAB desktop. Linux Holoscan runs under WSL2 / Ubuntu 22.04 and cannot load Windows PE/DLL dynamic libraries.
2. **MEX API Coupling:** Inspection of generated source code (`resnetPredict.cu`, `resnetPredict_mexutil.cu`, `callPredict.cu`) shows direct inclusion of MATLAB C/C++ MEX API headers (`mex.h`, `matrix.h`).
3. **Viable Linux Deployment Route:** To leverage MATLAB GPU Coder for Linux Holoscan C++ operators in future tasks (e.g. image enhancement or signal processing):
   - Configure MATLAB GPU Coder targeting **C++ Static/Dynamic Library (`coder.config('lib')`)** or standalone executable rather than MEX (`coder.config('mex')`).
   - Cross-compile generated CUDA C++ sources (`.cu`/`.cpp`) using `nvcc`/`g++` inside WSL Ubuntu-22.04.
