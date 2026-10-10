function [bModeImage, envelopeImg, timingMs] = reconstructDAS(rfData, params, targetSize, dynamicRangedB)
% RECONSTRUCTDAS Delay-and-Sum (DAS) beamforming & B-mode ultrasound image reconstruction.
%
% Syntax:
%   [bModeImage, envelopeImg, timingMs] = reconstructDAS(rfData, params, targetSize, dynamicRangedB)
%
% Inputs:
%   rfData         - (numSamples x numElements) raw RF channel data
%   params         - struct with fields: c, fs, f0, numElements, elementPositionsX, timeVector
%   targetSize     - [Nz, Nx] grid dimensions, default [224, 224]
%   dynamicRangedB - Dynamic range for log compression in dB, default 50
%
% Outputs:
%   bModeImage  - (Nz x Nx x 3) uint8 B-mode image ready for ResNet-18
%   envelopeImg - (Nz x Nx) double linear envelope values
%   timingMs    - Reconstruction runtime in milliseconds

    if nargin < 3 || isempty(targetSize)
        targetSize = [224, 224];
    end
    if nargin < 4 || isempty(dynamicRangedB)
        dynamicRangedB = 50;
    end

    Nz = targetSize(1);
    Nx = targetSize(2);

    tic;

    % 1. Spatial Reconstruction Grid (x, z)
    c = params.c;
    fs = params.fs;
    elemX = params.elementPositionsX;
    t = params.timeVector;
    numSamples = length(t);

    zMin = 5.0e-3;
    zMax = 45.0e-3;
    xMin = min(elemX);
    xMax = max(elemX);

    gridZ = linspace(zMin, zMax, Nz);
    gridX = linspace(xMin, xMax, Nx);

    [gridX2D, gridZ2D] = meshgrid(gridX, gridZ);

    % 2. Delay-and-Sum (DAS) Beamforming
    beamformedRF = zeros(Nz, Nx);

    % For each array element, calculate time delay for all grid points
    for elem = 1:params.numElements
        rxX = elemX(elem);

        % Round-trip travel time: plane wave tx (depth z) + spherical rx to element at rxX
        tau = (gridZ2D + sqrt((gridX2D - rxX).^2 + gridZ2D.^2)) / c;

        % Convert delay to fractional sample index
        sampleIndices = tau * fs + 1;

        % Bilinear / 1D linear interpolation of RF channel signal
        validMask = (sampleIndices >= 1) & (sampleIndices <= numSamples);
        elemRF = zeros(Nz, Nx);
        elemRF(validMask) = interp1(1:numSamples, rfData(:, elem), sampleIndices(validMask), 'linear', 0);

        beamformedRF = beamformedRF + elemRF;
    end

    % 3. Analytic Signal & Envelope Detection (FFT-based Hilbert Transform)
    envelopeImg = zeros(Nz, Nx);
    for col = 1:Nx
        scanline = beamformedRF(:, col);
        N = length(scanline);
        fftLine = fft(scanline);
        H = zeros(N, 1);
        if mod(N, 2) == 0
            H(1) = 1; H(N/2+1) = 1;
            H(2 : N/2) = 2;
        else
            H(1) = 1;
            H(2 : (N+1)/2) = 2;
        end
        analyticLine = ifft(fftLine .* H);
        envelopeImg(:, col) = abs(analyticLine);
    end

    % 4. Log Compression & Dynamic Range Mapping
    maxEnv = max(envelopeImg(:));
    if maxEnv == 0
        maxEnv = 1e-12;
    end

    % 20 * log10 scale normalized to peak
    logImg = 20 * log10((envelopeImg + 1e-12) / maxEnv);

    % Clamp to dynamic range [-dynamicRangedB, 0]
    normImg = (logImg + dynamicRangedB) / dynamicRangedB;
    normImg(normImg < 0) = 0;
    normImg(normImg > 1) = 1;

    % Convert to 8-bit grayscale
    gray8 = uint8(round(normImg * 255));

    % Replicate to 3 identical channels for 224x224x3 interface compatibility
    bModeImage = cat(3, gray8, gray8, gray8);

    timingMs = toc * 1000;
end
