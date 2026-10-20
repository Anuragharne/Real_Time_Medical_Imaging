# Ultrasound Image Enhancement Evaluation Report (CLAHE)

**Date:** October 10, 2026  
**Candidate Method:** Contrast Limited Adaptive Histogram Equalization (CLAHE / `adapthisteq`)  
**Evaluated Dataset Samples:** 3 representative BUSI images (`benign (1).png`, `malignant (1).png`, `normal (1).png`)  
**Classifier Model:** Fine-tuned ResNet-18 Breast Ultrasound Classifier (`models/netResNet.mat`)  
**Final Recommendation:** **REJECT CLAHE as an active input transformation in the inference pipeline** (retain unenhanced B-mode baseline).

---

## 1. Executive Summary

CLAHE was evaluated on three representative BUSI ultrasound images (`benign (1).png`, `malignant (1).png`, `normal (1).png`) to determine whether adaptive contrast enhancement improves image quality without degrading classification accuracy.

- **Visual / Contrast Impact:** CLAHE increases overall image contrast (standard deviation of pixel intensities increased from $\sim 58$ to $\sim 64$), enhancing boundary visibility to human observers.
- **Classification Impact:** CLAHE caused a **catastrophic misclassification** on the malignant sample. The true malignant sample (`malignant (1).png`), which is correctly classified as **Malignant** with $99.94\%$ confidence on unenhanced input, was misclassified as **Benign** ($63.97\%$ confidence) under standard CLAHE (`ClipLimit = 0.02`).
- **Parameter Sensitivity:** Testing clip limits from $0.005$ to $0.050$ confirmed that even minimal contrast equalization ($0.005$) drops malignant confidence from $99.94\%$ down to $2.30\%$, misclassifying the malignant sample.
- **Recommendation:** **REJECT CLAHE.** The fine-tuned ResNet-18 model relies on acoustic speckle patterns and hypoechoic attenuation characteristics present in native B-mode images. CLAHE distorts these critical feature representations.

---

## 2. Evaluation Setup & Parameters

- **Function:** MATLAB `adapthisteq` (OpenCV `cv::createCLAHE`)
- **Evaluated Clip Limits:** $0.005, 0.010, 0.015, 0.020, 0.030, 0.050$
- **Tile Grid Size:** $[8 \times 8]$ tiles
- **Pipeline Order:** Raw B-mode Image $\rightarrow$ Grayscale $\rightarrow$ CLAHE $\rightarrow$ Bicubic Resize ($224 \times 224$) $\rightarrow 3$-Channel Replicate $\rightarrow$ ResNet-18 Inference

---

## 3. Comparative Test Results (Standard CLAHE: ClipLimit = 0.02)

| Sample Image | True Class | Unenhanced Baseline Prediction & Scores `[Benign, Malignant, Normal]` | CLAHE Enhanced Prediction & Scores `[Benign, Malignant, Normal]` | Target Score Delta | Classification Status |
|---|---|---|---|---|---|
| `benign (1).png` | Benign | **Benign** `[0.999983, 0.000015, 0.000002]` | **Benign** `[0.999738, 0.000000, 0.000262]` | $-0.000245$ | **MATCH** (Correct) |
| `malignant (1).png` | Malignant | **Malignant** `[0.000461, 0.999420, 0.000119]` | **Benign** `[0.639653, 0.000327, 0.360020]` | **$-0.999092$** | **FAILED (False Negative)** |
| `normal (1).png` | Normal | **Normal** `[0.000161, 0.000001, 0.999839]` | **Normal** `[0.003563, 0.000026, 0.996411]` | $-0.003428$ | **MATCH** (Correct) |

---

## 4. CLAHE Clip Limit Sensitivity Analysis (`malignant (1).png`)

| Configuration | Clip Limit | Predicted Class | Benign Score | Malignant Score | Normal Score | Impact on Malignant Lesion |
|---|---|---|---|---|---|---|
| **Baseline (Unenhanced)** | None | **Malignant** | 0.000461 | **0.999420** | 0.000119 | True positive ($99.94\%$ confidence) |
| CLAHE | $0.005$ | **Benign** | 0.837898 | 0.022961 | 0.139142 | Flipped to Benign ($83.79\%$ confidence) |
| CLAHE | $0.010$ | **Normal** | 0.367429 | 0.000444 | 0.632127 | Flipped to Normal ($63.21\%$ confidence) |
| CLAHE | $0.015$ | **Normal** | 0.416411 | 0.000184 | 0.583404 | Flipped to Normal ($58.34\%$ confidence) |
| CLAHE | $0.020$ | **Benign** | 0.639653 | 0.000327 | 0.360020 | Flipped to Benign ($63.97\%$ confidence) |
| CLAHE | $0.030$ | **Benign** | 0.765172 | 0.000260 | 0.234568 | Flipped to Benign ($76.52\%$ confidence) |
| CLAHE | $0.050$ | **Benign** | 0.750286 | 0.000170 | 0.249544 | Flipped to Benign ($75.03\%$ confidence) |

---

## 5. Engineering Rationale & Next Steps

1. **Root Cause Analysis:** ResNet-18 features learned during fine-tuning correspond directly to native B-mode ultrasound intensity distributions, acoustic attenuation shadows, and speckle noise. CLAHE's local contrast stretching alters these low-level feature representations, destroying acoustic attenuation cues vital for malignant diagnosis.
2. **Production Pipeline Decision:** Do **not** integrate CLAHE into `holoscan/app/main.cpp`. The C++ Holoscan pipeline will maintain the validated unenhanced B-mode image preprocessing path (`cv::imread` $\rightarrow$ `COLOR_BGR2GRAY` $\rightarrow$ `cv::INTER_CUBIC` resize).
3. **Preserved Fallback:** The unenhanced baseline remains 100% operational, fully verified, and ready for RF-reconstruction integration in subsequent checkpoints.

---

## 6. Subsequent Alternative Enhancement Evaluation (Wiener Filter)

In Checkpoint B, a 2D adaptive Wiener filter (`wiener2`) was evaluated as an alternative enhancement candidate on the same three BUSI samples. While Wiener filtering avoided flipping classification labels (unlike CLAHE), it decreased malignant diagnostic confidence by $0.11\%$ (for $[3 \times 3]$) and $5.60\%$ (for $[5 \times 5]$) due to boundary softening without yielding clinical benefit. Consequently, both CLAHE and Wiener filtering are formally rejected in favor of retaining the unenhanced native B-mode baseline. Full experimental data and metrics are detailed in [checkpoint_b_enhancement_gpu_coder.md](file:///c:/Users/anura/Desktop/RealTimeMedicalImaging/docs/checkpoint_b_enhancement_gpu_coder.md).

