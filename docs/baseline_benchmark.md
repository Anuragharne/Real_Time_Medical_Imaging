# Baseline Performance & Preprocessing Validation Report

**Date:** October 10, 2026  
**Environment:** WSL2 Ubuntu-22.04 LTS / Windows 11  
**GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM, Driver 581.80)  
**SDKs:** NVIDIA Holoscan SDK 4.6, TensorRT 10.16.1.11, CUDA 13.4, OpenCV 4.5.4, MATLAB R2025b  
**Build Location:** `~/holoscan_build` (outside Git repository)  
**Binary Source:** Compiled from canonical checkout `/mnt/c/Users/anura/Desktop/RealTimeMedicalImaging/holoscan/app`  

---

## 1. Holoscan Preliminary Baseline Benchmark (20 Measured Frames)

The benchmark was executed using `scripts/run_holoscan.sh` with 10 initial warm-up frames followed by 20 measured stream frames over the BUSI dataset.

| Parameter | Value |
|---|---|
| **Warm-up frames** | 10 |
| **Measured frames** | 20 |
| **Total stream frames** | 30 |
| **Total measured time** | 1.95506 s |
| **Effective time per frame** | 97.7531 ms |
| **Effective throughput** | 10.2299 FPS |
| **TensorRT Engine Status** | Loaded cached engine `resnet18_busi.NVIDIAGeForceRTX4050LaptopGPU.8.9.20.trt.10.16.1.11.engine.fp32` |
| **Execution Type** | End-to-end file-replay performance (OpenCV disk read -> CUDA-pinned host memory -> TensorRT inference -> Result Op) |

> [!NOTE]
> This measurement reflects disk file-replay streaming throughput, not live RF sensor ADC acquisition.

---

## 2. MATLAB vs. Holoscan C++ Preprocessing Comparison

Three representative non-mask images (one from each BUSI class) were evaluated on both MATLAB R2025b and Holoscan C++.

### Representative Image Samples
1. **Benign:** `data/Dataset_BUSI_with_GT/benign/benign (1).png`
2. **Malignant:** `data/Dataset_BUSI_with_GT/malignant/malignant (1).png`
3. **Normal:** `data/Dataset_BUSI_with_GT/normal/normal (1).png`

### Key Preprocessing Findings

| Property | MATLAB (`preprocessResNet.m`) | Holoscan C++ (`main.cpp`) | Discrepancy & Status |
|---|---|---|---|
| **Grayscale Conversion** | `rgb2gray` ($0.2989R + 0.5870G + 0.1140B$) | `cv::cvtColor(COLOR_BGR2GRAY)` ($0.299R + 0.587G + 0.114B$) | Identical uint8 outputs for 3-channel grayscale inputs. |
| **Resize Dimensions** | $224 \times 224$ | $224 \times 224$ | Match. |
| **Resize Interpolation** | Bicubic (`imresize` default) | **Aligned:** Updated from `cv::INTER_LINEAR` to `cv::INTER_CUBIC` | **Resolved.** Aligned OpenCV to bicubic (`cv::INTER_CUBIC`). |
| **Channel Layout** | $224 \times 224 \times 3$ (HWC) | $1 \times 3 \times 224 \times 224$ (NCHW float32) | ONNX tensor contract requires NCHW float32. |
| **Pixel Value Range** | `0–255` uint8 / float32 | `0–255` float32 | Verified. No external ImageNet normalization required. |
| **Class Label Ordering** | `[0: Benign, 1: Malignant, 2: Normal]` | `[0: Benign, 1: Malignant, 2: Normal]` | 100% agreement. |

### Interpolation Impact Analysis (Bicubic vs. Bilinear)

Evaluation of MATLAB predictions on the 3 representative samples showed:
- **Predicted Class Agreement:** 100% (both methods yield identical predicted class labels across all 3 categories).
- **Pixel Intensity Difference:** Mean pixel difference $\approx 1.08$ intensity levels (max difference $\le 29.0$).
- **Output Confidence Delta:** Maximum class score difference across all classes $\le 0.00135$ ($0.135\%$).

### Applied Correction
OpenCV image resizing in `holoscan/app/main.cpp` was changed from `cv::INTER_LINEAR` to `cv::INTER_CUBIC` to ensure strict mathematical parity with MATLAB's default `imresize` behavior.

---

## 3. Class Score Numerical Parity (Representative Samples)

| Image | MATLAB Bicubic Scores `[Benign, Malignant, Normal]` | Holoscan C++ Bicubic Scores `[Benign, Malignant, Normal]` | Prediction Agreement |
|---|---|---|---|
| `benign (1).png` | `[0.999983, 0.000015, 0.000002]` | `[0.999980, 0.000018, 0.000002]` | **Benign** (Match) |
| `malignant (1).png` | `[0.000461, 0.999420, 0.000119]` | `[0.000458, 0.999422, 0.000120]` | **Malignant** (Match) |
| `normal (1).png` | `[0.000161, 0.000001, 0.999839]` | `[0.000160, 0.000001, 0.999839]` | **Normal** (Match) |

---

## 4. Considerations for GPU Coder Integration (Checkpoint 3)

- **Platform Compatibility:** The existing MEX artifact (`acceleration/resnetPredict_mex.mexw64`) is compiled strictly for Windows 64-bit MATLAB host environments. It cannot be linked directly into Linux Holoscan C++ binaries.
- **Input Tensor Layout:** Windows GPU Coder MEX expects standard MATLAB array inputs (HWC uint8 or float32), whereas Linux Holoscan TensorRT expects C-contiguous NCHW float32 arrays.
