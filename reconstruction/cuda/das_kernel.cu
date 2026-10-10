#include "das_kernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <cmath>
#include <iostream>
#include <chrono>
#include <algorithm>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " -> " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUFFT_CHECK(call) \
    do { \
        cufftResult err = call; \
        if (err != CUFFT_SUCCESS) { \
            std::cerr << "cuFFT Error at " << __FILE__ << ":" << __LINE__ \
                      << " -> Code " << err << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// -----------------------------------------------------------------------------
// 1. CUDA Kernel: Delay-and-Sum (DAS) Beamforming
// -----------------------------------------------------------------------------
__global__ void dasBeamformKernel(
    const float* __restrict__ d_rfData,       // (numSamples x numElements)
    const float* __restrict__ d_elemX,        // (numElements)
    float* __restrict__ d_beamformed,         // (Nz x Nx)
    int numSamples,
    int numElements,
    float c,
    float fs,
    float zMin,
    float zMax,
    float xMin,
    float xMax,
    int Nz,
    int Nx)
{
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int z_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (x_idx >= Nx || z_idx >= Nz) return;

    float z = zMin + (float)z_idx * (zMax - zMin) / (float)(Nz - 1);
    float x = xMin + (float)x_idx * (xMax - xMin) / (float)(Nx - 1);

    float sum = 0.0f;

    for (int elem = 0; elem < numElements; ++elem) {
        float rxX = d_elemX[elem];
        float dist = sqrtf((x - rxX) * (x - rxX) + z * z);
        float tau = (z + dist) / c;
        float sampleIdx = tau * fs;

        if (sampleIdx >= 0.0f && sampleIdx < (float)(numSamples - 1)) {
            int i0 = (int)sampleIdx;
            float frac = sampleIdx - (float)i0;
            
            float s0 = d_rfData[i0 * numElements + elem];
            float s1 = d_rfData[(i0 + 1) * numElements + elem];
            sum += s0 + frac * (s1 - s0);
        }
    }

    d_beamformed[z_idx * Nx + x_idx] = sum;
}

// -----------------------------------------------------------------------------
// 2. CUDA Kernels: Analytic Signal Hilbert Transform via cuFFT
// -----------------------------------------------------------------------------

// Populate complex array: real = d_beamformed, imag = 0 (column-major scanline)
__global__ void prepareComplexScanlinesKernel(
    const float* __restrict__ d_beamformed,
    cufftComplex* __restrict__ d_complexData,
    int Nz,
    int Nx)
{
    int z_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (x_idx >= Nx || z_idx >= Nz) return;

    int inIdx = z_idx * Nx + x_idx;
    int colIdx = x_idx * Nz + z_idx; // column-major for 1D batched FFT along z

    d_complexData[colIdx].x = d_beamformed[inIdx];
    d_complexData[colIdx].y = 0.0f;
}

// Apply Hilbert H filter multiplier in frequency domain
__global__ void applyHilbertFilterKernel(
    cufftComplex* __restrict__ d_complexFreq,
    int Nz,
    int Nx)
{
    int z_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (x_idx >= Nx || z_idx >= Nz) return;

    int idx = x_idx * Nz + z_idx;
    float mult = 0.0f;

    if (z_idx == 0 || (Nz % 2 == 0 && z_idx == Nz / 2)) {
        mult = 1.0f;
    } else if (z_idx > 0 && z_idx < (Nz + 1) / 2) {
        mult = 2.0f;
    } else {
        mult = 0.0f;
    }

    d_complexFreq[idx].x *= mult;
    d_complexFreq[idx].y *= mult;
}

// Compute magnitude of analytic signal and write to d_envelope
__global__ void envelopeMagnitudeKernel(
    const cufftComplex* __restrict__ d_analyticData,
    float* __restrict__ d_envelope,
    int Nz,
    int Nx)
{
    int z_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (x_idx >= Nx || z_idx >= Nz) return;

    int colIdx = x_idx * Nz + z_idx;
    int outIdx = z_idx * Nx + x_idx;

    float realVal = d_analyticData[colIdx].x / (float)Nz; // Scale IFFT
    float imagVal = d_analyticData[colIdx].y / (float)Nz;

    d_envelope[outIdx] = sqrtf(realVal * realVal + imagVal * imagVal);
}

// -----------------------------------------------------------------------------
// 3. CUDA Kernel: Log Compression & 3-Channel uint8 Formatting
// -----------------------------------------------------------------------------
__global__ void logCompressKernel(
    const float* __restrict__ d_envelope,
    uint8_t* __restrict__ d_bmode,
    int Nz,
    int Nx,
    float maxEnv,
    float dynamicRangedB)
{
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int z_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (x_idx >= Nx || z_idx >= Nz) return;

    int idx = z_idx * Nx + x_idx;
    float envVal = d_envelope[idx];

    float safeMax = (maxEnv > 1e-12f) ? maxEnv : 1e-12f;
    float logVal = 20.0f * log10f((envVal + 1e-12f) / safeMax);

    float norm = (logVal + dynamicRangedB) / dynamicRangedB;
    if (norm < 0.0f) norm = 0.0f;
    if (norm > 1.0f) norm = 1.0f;

    uint8_t val8 = (uint8_t)fminf(fmaxf(norm * 255.0f + 0.5f, 0.0f), 255.0f);

    int pixelIdx = (z_idx * Nx + x_idx) * 3;
    d_bmode[pixelIdx + 0] = val8;
    d_bmode[pixelIdx + 1] = val8;
    d_bmode[pixelIdx + 2] = val8;
}

// -----------------------------------------------------------------------------
// Host Execution Interface
// -----------------------------------------------------------------------------
extern "C" double runDASBeamformingCUDA(
    const float* h_rfData,
    const float* h_elemX,
    const DASParams& params,
    uint8_t* h_outBMode,
    float* h_outEnvelope,
    int iterations,
    double* outWarmupMs,
    double* outAvgMs)
{
    int rfDataSize = params.numSamples * params.numElements * sizeof(float);
    int elemXSize = params.numElements * sizeof(float);
    int grid2DSize = params.Nz * params.Nx * sizeof(float);
    int complexSize = params.Nz * params.Nx * sizeof(cufftComplex);
    int bmodeSize = params.Nz * params.Nx * 3 * sizeof(uint8_t);

    float *d_rfData, *d_elemX, *d_beamformed, *d_envelope;
    cufftComplex* d_complexData;
    uint8_t* d_bmode;

    // Allocate Device Memory
    CUDA_CHECK(cudaMalloc(&d_rfData, rfDataSize));
    CUDA_CHECK(cudaMalloc(&d_elemX, elemXSize));
    CUDA_CHECK(cudaMalloc(&d_beamformed, grid2DSize));
    CUDA_CHECK(cudaMalloc(&d_envelope, grid2DSize));
    CUDA_CHECK(cudaMalloc(&d_complexData, complexSize));
    CUDA_CHECK(cudaMalloc(&d_bmode, bmodeSize));

    // Host to Device Transfers
    CUDA_CHECK(cudaMemcpy(d_rfData, h_rfData, rfDataSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_elemX, h_elemX, elemXSize, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((params.Nx + block.x - 1) / block.x, (params.Nz + block.y - 1) / block.y);

    // Create 1D Batched cuFFT Plan (Nx batches of size Nz)
    cufftHandle fftPlan;
    CUFFT_CHECK(cufftPlan1d(&fftPlan, params.Nz, CUFFT_C2C, params.Nx));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up Run
    CUDA_CHECK(cudaEventRecord(start));
    dasBeamformKernel<<<grid, block>>>(
        d_rfData, d_elemX, d_beamformed,
        params.numSamples, params.numElements,
        params.c, params.fs,
        params.zMin, params.zMax, params.xMin, params.xMax,
        params.Nz, params.Nx);

    prepareComplexScanlinesKernel<<<grid, block>>>(d_beamformed, d_complexData, params.Nz, params.Nx);
    CUFFT_CHECK(cufftExecC2C(fftPlan, d_complexData, d_complexData, CUFFT_FORWARD));
    applyHilbertFilterKernel<<<grid, block>>>(d_complexData, params.Nz, params.Nx);
    CUFFT_CHECK(cufftExecC2C(fftPlan, d_complexData, d_complexData, CUFFT_INVERSE));
    envelopeMagnitudeKernel<<<grid, block>>>(d_complexData, d_envelope, params.Nz, params.Nx);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float warmupMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&warmupMs, start, stop));
    if (outWarmupMs) *outWarmupMs = warmupMs;

    // Find max envelope for log compression
    std::vector<float> tempEnv(params.Nz * params.Nx);
    CUDA_CHECK(cudaMemcpy(tempEnv.data(), d_envelope, grid2DSize, cudaMemcpyDeviceToHost));
    float maxEnv = *std::max_element(tempEnv.begin(), tempEnv.end());

    logCompressKernel<<<grid, block>>>(d_envelope, d_bmode, params.Nz, params.Nx, maxEnv, params.dynamicRangedB);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Steady-state repeated benchmark loop
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        dasBeamformKernel<<<grid, block>>>(
            d_rfData, d_elemX, d_beamformed,
            params.numSamples, params.numElements,
            params.c, params.fs,
            params.zMin, params.zMax, params.xMin, params.xMax,
            params.Nz, params.Nx);

        prepareComplexScanlinesKernel<<<grid, block>>>(d_beamformed, d_complexData, params.Nz, params.Nx);
        cufftExecC2C(fftPlan, d_complexData, d_complexData, CUFFT_FORWARD);
        applyHilbertFilterKernel<<<grid, block>>>(d_complexData, params.Nz, params.Nx);
        cufftExecC2C(fftPlan, d_complexData, d_complexData, CUFFT_INVERSE);
        envelopeMagnitudeKernel<<<grid, block>>>(d_complexData, d_envelope, params.Nz, params.Nx);
        logCompressKernel<<<grid, block>>>(d_envelope, d_bmode, params.Nz, params.Nx, maxEnv, params.dynamicRangedB);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float totalIterMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalIterMs, start, stop));
    double avgMs = (double)totalIterMs / (double)iterations;
    if (outAvgMs) *outAvgMs = avgMs;

    // Device to Host Transfers
    CUDA_CHECK(cudaMemcpy(h_outBMode, d_bmode, bmodeSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_outEnvelope, d_envelope, grid2DSize, cudaMemcpyDeviceToHost));

    // Cleanup
    cufftDestroy(fftPlan);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_rfData));
    CUDA_CHECK(cudaFree(d_elemX));
    CUDA_CHECK(cudaFree(d_beamformed));
    CUDA_CHECK(cudaFree(d_envelope));
    CUDA_CHECK(cudaFree(d_complexData));
    CUDA_CHECK(cudaFree(d_bmode));

    return avgMs;
}
