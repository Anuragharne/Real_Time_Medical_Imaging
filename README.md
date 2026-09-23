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