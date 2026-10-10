#ifndef DAS_KERNEL_H
#define DAS_KERNEL_H

#include <cstdint>
#include <vector>

struct DASParams {
    float c;                 // Speed of sound (m/s)
    float fs;                // Sampling frequency (Hz)
    float f0;                // Center frequency (Hz)
    int numElements;         // Number of array elements (64)
    int numSamples;          // Number of time samples (2338)
    int Nz;                  // Image height (224)
    int Nx;                  // Image width (224)
    float zMin;              // Min depth (m)
    float zMax;              // Max depth (m)
    float xMin;              // Min lateral (m)
    float xMax;              // Max lateral (m)
    float dynamicRangedB;    // Dynamic range (50 dB)
};

#ifdef __cplusplus
extern "C" {
#endif

// Main CUDA execution interface
double runDASBeamformingCUDA(
    const float* h_rfData,       // (numSamples x numElements) row-major
    const float* h_elemX,        // (numElements)
    const DASParams& params,
    uint8_t* h_outBMode,         // (Nz x Nx x 3) output uint8
    float* h_outEnvelope,        // (Nz x Nx) output float
    int iterations,
    double* outWarmupMs,
    double* outAvgMs
);

#ifdef __cplusplus
}
#endif

#endif // DAS_KERNEL_H
