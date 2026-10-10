#include "das_kernel.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>

int main(int argc, char** argv) {
    std::string rfBinPath = "rf_data.bin";
    std::string paramsBinPath = "params.bin";
    std::string outBModePath = "bmode_cuda.bin";
    std::string outEnvPath = "envelope_cuda.bin";

    if (argc > 1) rfBinPath = argv[1];
    if (argc > 2) paramsBinPath = argv[2];
    if (argc > 3) outBModePath = argv[3];
    if (argc > 4) outEnvPath = argv[4];

    std::cout << "==========================================================" << std::endl;
    std::cout << " CUDA Delay-and-Sum (DAS) Ultrasound Reconstruction" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // Load Params
    std::ifstream pFile(paramsBinPath, std::ios::binary);
    if (!pFile.is_open()) {
        std::cerr << "Error: Could not open params binary file: " << paramsBinPath << std::endl;
        return 1;
    }
    DASParams params;
    pFile.read(reinterpret_cast<char*>(&params), sizeof(DASParams));
    pFile.close();

    std::cout << "Parameters Loaded:" << std::endl;
    std::cout << "  Sound Speed (c)   : " << params.c << " m/s" << std::endl;
    std::cout << "  Sampling Freq (fs): " << params.fs / 1e6 << " MHz" << std::endl;
    std::cout << "  Elements          : " << params.numElements << std::endl;
    std::cout << "  RF Samples/Channel: " << params.numSamples << std::endl;
    std::cout << "  Grid Size (Nz x Nx): " << params.Nz << " x " << params.Nx << std::endl;

    // Load RF Data
    std::ifstream rfFile(rfBinPath, std::ios::binary);
    if (!rfFile.is_open()) {
        std::cerr << "Error: Could not open RF binary file: " << rfBinPath << std::endl;
        return 1;
    }
    int totalRFSamples = params.numSamples * params.numElements;
    std::vector<float> h_rfData(totalRFSamples);
    rfFile.read(reinterpret_cast<char*>(h_rfData.data()), totalRFSamples * sizeof(float));
    rfFile.close();

    // Element Positions
    std::vector<float> h_elemX(params.numElements);
    for (int i = 0; i < params.numElements; ++i) {
        h_elemX[i] = params.xMin + i * (params.xMax - params.xMin) / (params.numElements - 1);
    }

    std::vector<uint8_t> h_outBMode(params.Nz * params.Nx * 3);
    std::vector<float> h_outEnvelope(params.Nz * params.Nx);

    double warmupMs = 0.0;
    double avgMs = 0.0;
    int benchmarkIters = 50;

    std::cout << "\nExecuting CUDA Beamforming (" << benchmarkIters << " iterations)..." << std::endl;
    runDASBeamformingCUDA(
        h_rfData.data(),
        h_elemX.data(),
        params,
        h_outBMode.data(),
        h_outEnvelope.data(),
        benchmarkIters,
        &warmupMs,
        &avgMs
    );

    std::cout << "==========================================================" << std::endl;
    std::cout << " CUDA Performance Metrics:" << std::endl;
    std::cout << "   Warm-up Latency       : " << std::fixed << std::setprecision(3) << warmupMs << " ms" << std::endl;
    std::cout << "   Steady-State Latency  : " << std::fixed << std::setprecision(3) << avgMs << " ms / frame" << std::endl;
    std::cout << "   Steady-State Throughput: " << std::fixed << std::setprecision(2) << (1000.0 / avgMs) << " FPS" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // Save outputs
    std::ofstream bmFile(outBModePath, std::ios::binary);
    if (bmFile.is_open()) {
        bmFile.write(reinterpret_cast<const char*>(h_outBMode.data()), h_outBMode.size() * sizeof(uint8_t));
        bmFile.close();
        std::cout << "Saved CUDA B-mode output to: " << outBModePath << std::endl;
    }

    std::ofstream envFile(outEnvPath, std::ios::binary);
    if (envFile.is_open()) {
        envFile.write(reinterpret_cast<const char*>(h_outEnvelope.data()), h_outEnvelope.size() * sizeof(float));
        envFile.close();
        std::cout << "Saved CUDA Envelope output to: " << outEnvPath << std::endl;
    }

    return 0;
}
