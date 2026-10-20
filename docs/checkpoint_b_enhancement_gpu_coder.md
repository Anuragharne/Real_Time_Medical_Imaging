# Checkpoint B Report: Enhancement Evaluation & GPU Coder Feasibility

**Date:** October 20, 2026  
**Status:** Complete (Checkpoint B)  
**Evaluated Components:**
1. Wiener Filter Ultrasound Image Enhancement (`wiener2`)
2. Standalone Linux GPU Coder Code Generation & Toolchain Feasibility

---

## 1. Executive Summary & Decisions

| Evaluation Track | Primary Finding | Architectural Decision |
|---|---|---|
| **Alternative Enhancement (Wiener Filter)** | Preserves correct classification labels on all 3 representative BUSI test samples (`benign`, `malignant`, `normal`), but reduces correct-class confidence on the malignant lesion (from $99.94\%$ to $99.83\%$ for $[3 \times 3]$ and $94.34\%$ for $[5 \times 5]$) due to boundary softening. | **REJECT Wiener filter as an active transformation; retain the unenhanced B-mode baseline.** Native ResNet-18 features are tuned to unenhanced speckle and attenuation patterns. |
| **Linux GPU Coder Feasibility** | Standalone CUDA C++ source generation (`coder.gpuConfig('lib')` with `GenCodeOnly = true`) successfully generates portable C++/CUDA code decoupled from MEX APIs. The generated sources were compiled under Linux WSL2 (Ubuntu 22.04, CUDA 13.4) into a static library (`libgpusimpletest.a`, 25.8 KB). | **RETAIN native CUDA C++ (`das_kernel.cu`) and TensorRT engine (`sample_resnet18.engine`) for Holoscan deployment.** Windows GPU Coder MEX is retained for Windows MATLAB acceleration. Forcing generated code into Holoscan offers no performance or architectural benefit. |

---

## 2. Track 1: Wiener Filter Image Enhancement Evaluation

### 2.1 Motivation & Context
In Checkpoint 4, Contrast Limited Adaptive Histogram Equalization (CLAHE) was evaluated and rejected because contrast stretching caused a catastrophic misclassification on `malignant (1).png` (flipping the diagnosis to Benign with $63.97\%$ confidence). 

To test whether edge-preserving noise reduction provides benefit without distorting attenuation signatures, a **Wiener filter** (`wiener2`) was evaluated. Wiener filtering performs 2D adaptive low-pass filtering based on local mean and variance estimates:
$$\mu = \frac{1}{NM}\sum_{p,q \in \eta} a(p,q), \quad \sigma^2 = \frac{1}{NM}\sum_{p,q \in \eta} a^2(p,q) - \mu^2$$
$$b(n_1,n_2) = \mu + \frac{\sigma^2 - \nu^2}{\sigma^2} (a(n_1,n_2) - \mu)$$
where $\nu^2$ is the estimated noise variance.

### 2.2 Test Configuration
- **Model:** Fine-tuned ResNet-18 (`models/netResNet.mat`), frozen weights (no retraining).
- **Test Samples:**
  1. `data/Dataset_BUSI_with_GT/benign/benign (1).png`
  2. `data/Dataset_BUSI_with_GT/malignant/malignant (1).png`
  3. `data/Dataset_BUSI_with_GT/normal/normal (1).png`
- **Filter Configurations:**
  - Window size $[3 \times 3]$ (mild speckle smoothing)
  - Window size $[5 \times 5]$ (aggressive speckle smoothing)
- **Processing Order:** Raw Image $\rightarrow$ Grayscale $\rightarrow$ Wiener Filter $\rightarrow$ Bicubic Resize ($224 \times 224$) $\rightarrow 3$-Channel Replicate $\rightarrow$ ResNet-18 Inference.

### 2.3 Quantitative Results

| Sample Image | Filter Setting | Predicted Class | Benign Score | Malignant Score | Normal Score | Target Score Delta | PSNR (dB) | MAE (0–255) | Intensity StdDev |
|---|---|---|---|---|---|---|---|---|---|
| **`benign (1).png`** | Baseline (None) | **Benign** | **0.999983** | 0.000015 | 0.000002 | Baseline | — | — | 58.39 |
| | Wiener $[3 \times 3]$ | **Benign** | **0.999987** | 0.000012 | 0.000001 | $+0.000004$ | 37.86 | 2.41 | 57.91 |
| | Wiener $[5 \times 5]$ | **Benign** | **0.999981** | 0.000017 | 0.000002 | $-0.000002$ | 32.72 | 4.48 | 57.47 |
| **`malignant (1).png`** | Baseline (None) | **Malignant** | 0.000461 | **0.999420** | 0.000119 | Baseline | — | — | 61.46 |
| | Wiener $[3 \times 3]$ | **Malignant** | 0.001474 | **0.998325** | 0.000201 | $-0.001094$ | 40.44 | 1.86 | 61.16 |
| | Wiener $[5 \times 5]$ | **Malignant** | 0.053773 | **0.943449** | 0.002778 | $-0.055971$ | 34.78 | 3.67 | 60.87 |
| **`normal (1).png`** | Baseline (None) | **Normal** | 0.000161 | 0.000001 | **0.999839** | Baseline | — | — | 52.03 |
| | Wiener $[3 \times 3]$ | **Normal** | 0.000859 | 0.000009 | **0.999132** | $-0.000706$ | 41.36 | 1.63 | 51.81 |
| | Wiener $[5 \times 5]$ | **Normal** | 0.011526 | 0.000679 | **0.987795** | $-0.012044$ | 36.68 | 2.88 | 51.61 |

### 2.4 Diagnostic & Engineering Assessment
1. **Label Retention:** Unlike CLAHE, Wiener filtering did not flip any diagnostic label. All three samples retained their correct classifications under both $[3 \times 3]$ and $[5 \times 5]$ settings.
2. **Confidence Degradation on Malignant Lesion:** As the filtering neighborhood expands to $[5 \times 5]$, the malignant class score drops from $99.94\%$ to $94.34\%$ (a $5.6\%$ reduction), while the benign false-positive score increases from $0.05\%$ to $5.38\%$. This occurs because spatial smoothing blurs the irregular, spicular acoustic boundaries characteristic of malignant carcinomas.
3. **Absence of Diagnostic Improvement:** Across all three samples, Wiener filtering yielded no improvement in diagnostic confidence or accuracy. Standard deviation of intensity slightly decreased ($\sim 58.39 \rightarrow 57.91$ for benign, $61.46 \rightarrow 61.16$ for malignant), reflecting speckle suppression without enhanced lesion contrast.
4. **Exploratory Scope Disclaimer:** This test represents a small exploratory evaluation on three representative BUSI samples, not proof of clinical efficacy or full dataset generalization. Reconstructed synthetic RF images represent out-of-domain inputs and must not be conflated with clinical diagnostic evidence.
5. **Conclusion:** **REJECT Wiener filtering.** Adding 2D spatial filtering adds per-frame computational latency without providing classification gains. The active Holoscan and MATLAB pipelines retain the unenhanced B-mode baseline.

---

## 3. Track 2: Standalone Linux GPU Coder Feasibility Assessment

### 3.1 Host & Target Environment Audit

| Property | Windows 11 Host | Linux Environment (WSL2 Ubuntu 22.04 LTS) |
|---|---|---|
| **MATLAB Installation** | MATLAB R2025b (v25.2 Update 7) | None (`which matlab` is empty) |
| **GPU Coder License** | Active (`license('test', 'GPU_Coder') = 1`) | N/A |
| **MATLAB Coder License** | Active (`license('test', 'MATLAB_Coder') = 1`) | N/A |
| **Parallel Computing** | Active (`license('test', 'Distrib_Computing_Toolbox') = 1`) | N/A |
| **C/C++ Host Compiler** | Microsoft Visual C++ 2022 (MSVC 17.0 Build Tools) | GCC / G++ 11.4.0 (`/usr/bin/g++`) |
| **CUDA Compiler** | NVIDIA CUDA Toolkit 12.1 (`nvcc` v12.1.66) | NVIDIA CUDA Toolkit 13.4 (`/usr/local/cuda-13.4/bin/nvcc`) |
| **Target GPU** | NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM) | Same GPU accessible via WSL2 Direct3D/CUDA driver |

### 3.2 Architectural Analysis & Technical Constraints
1. **Windows MEX Limitations (`.mexw64`):**
   - The existing `acceleration/resnetPredict_mex.mexw64` (709.6 KB) was built for Windows x86_64 using MSVC 2022 and CUDA 12.1.
   - It directly links against Windows MATLAB MEX APIs (`mex.h`, `matrix.h`, `libmx.dll`, `libmex.dll`).
   - It cannot be loaded or executed natively inside Linux ELF environments (such as Holoscan running in Ubuntu 22.04).
2. **Cross-Compilation Toolchain Constraints:**
   - Windows MATLAB lacks Embedded Coder hardware target packages (e.g., `coder.hardware('NVIDIA Jetson')` or generic Linux cross-toolchains). Direct cross-compilation of Linux `.so` shared libraries from Windows MATLAB via GUI/CLI is therefore unsupported out of the box.
3. **Decoupled Code Generation Workflow (Proof of Concept):**
   - GPU Coder can be configured to generate standalone C++ / CUDA source code decoupled from the MEX interface using:
     ```matlab
     cfg = coder.gpuConfig('lib');
     cfg.GenCodeOnly = true;
     ```
   - This exports raw `.cu` and `.h` files that rely only on standard C/C++ libraries and fundamental MATLAB scalar type headers (`rtwtypes.h`, `tmwtypes.h`), completely free of `mex.h`.

### 3.3 Standalone Linux Code Generation & Compilation Proof of Concept

#### Step 1: MATLAB Code Generation on Windows
- **Source Function:** `acceleration/gpuSimpleTest.m` (GPU kernel definition using `coder.gpu.kernelfun()`):
  ```matlab
  function y = gpuSimpleTest(x) %#codegen
  coder.gpu.kernelfun();
  y = x .* 2;
  end
  ```
- **Execution Command:**
  ```matlab
  cfg = coder.gpuConfig('lib');
  cfg.GenCodeOnly = true;
  codegen -config cfg acceleration/gpuSimpleTest.m -args {zeros(100,1,'single')} -d '<scratch>/gpu_coder_standalone_test'
  ```
- **Generated Source Files:**
  - `gpuSimpleTest.cu` (implements `void gpuSimpleTest(const float x[100], float y[100])`)
  - `gpuSimpleTest.h` (clean C++ header, no `mex.h`)
  - `gpuSimpleTest_data.cu`, `gpuSimpleTest_data.h`
  - `gpuSimpleTest_initialize.cu`, `gpuSimpleTest_initialize.h`
  - `gpuSimpleTest_terminate.cu`, `gpuSimpleTest_terminate.h`
  - `rtwtypes.h`

#### Step 2: Linux Native Compilation under WSL2
- **Compiler:** NVIDIA CUDA Compiler 13.4 (`/usr/local/cuda-13.4/bin/nvcc`) with GCC 11.4 host compiler.
- **Include Path:** `-I"/mnt/c/Program Files/MATLAB/R2025b/extern/include"` (for `tmwtypes.h`).
- **Compilation Command:**
  ```bash
  nvcc -c -Xcompiler -fPIC -I"/mnt/c/Program Files/MATLAB/R2025b/extern/include" \
      gpuSimpleTest.cu gpuSimpleTest_data.cu gpuSimpleTest_initialize.cu gpuSimpleTest_terminate.cu
  ar rcs libgpusimpletest.a *.o
  ```
- **Verification Evidence:**
  ```
  -rwxrwxrwx 1 anurag-h anurag-h 25768 Oct 20 11:26 libgpusimpletest.a
  Compilation and static library generation successful!
  ```
  The native Linux static library `libgpusimpletest.a` (25,768 bytes) was generated successfully without warnings.

### 3.4 Feasibility Assessment for Holoscan Production Pipeline

| Pipeline Stage | Current Production Implementation in Holoscan | GPU Coder Standalone Feasibility | Recommendation & Rationale |
|---|---|---|---|
| **RF DAS Beamforming** | Native CUDA C++ (`reconstruction/cuda/das_kernel.cu` + cuFFT) | Could generate C++ DAS code, but cuFFT envelope detection and dynamic delay loops would require complex coder scaffolding. | **RETAIN native CUDA C++ implementation.** Achieves $0.133\text{ ms}$ kernel execution, $73.04\text{ dB}$ B-mode PSNR, zero external MATLAB runtime dependencies. |
| **ResNet-18 Inference** | NVIDIA TensorRT FP32 Engine (`sample_resnet18.engine` via `nvinfer` / TensorRT C++ API) | GPU Coder deep learning export requires linking against host cuDNN or TensorRT during codegen. Generates extensive boilerplate (`MW*MemoryManager`). | **RETAIN native TensorRT implementation.** Delivers $1.56\text{ ms}$ inference latency, $7 \times 10^{-5}$ score parity with MATLAB, native GXF integration. |
| **Windows Desktop App** | Windows GPU Coder MEX (`resnetPredict_mex.mexw64`) | Already implemented and verified ($3.22\times$ speedup, $2.57\text{ ms}$ latency). | **RETAIN Windows MEX for MATLAB desktop app only.** |

**Conclusion:** Standalone Linux GPU Coder compilation is technically feasible for basic mathematical kernels, but forcing it into the production Holoscan application would add complexity and MATLAB header dependencies without offering latency or accuracy improvements over native CUDA and TensorRT.

---

## 4. Documentation Updates & Limitations

1. **Active Enhancement Pipeline:** Unenhanced B-mode input remains the official, validated pipeline baseline across all tools (`ultrasoundApp.m`, `predictUltrasound.m`, and `holoscan/app/main.cpp`).
2. **Measurement Integrity Notice:** Benchmark timings remain tied strictly to their documented test environments:
   - MATLAB CPU reconstruction: $\sim 541\text{ ms}$.
   - Native CUDA DAS kernel: $0.133\text{ ms}$ ($1.175\text{ ms}$ including PCIe transfers).
   - Holoscan Live RF Replay (`task-1016`): $8.73\text{ ms/frame}$ ($\sim 114.6\text{ FPS}$).
   - Historical cold/first-frame runs recorded up to $25.22\text{ ms/frame}$; results are not conflated.
