% MATLAB Harness: Export RF Data, Execute CUDA DAS, and Compare Parity & Latency
scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(scriptDir);
addpath(fullfile(repoRoot, 'reconstruction'));
addpath(fullfile(repoRoot, 'preprocessing'));

fprintf('==========================================================\n');
fprintf(' MATLAB CPU vs CUDA GPU Delay-and-Sum (DAS) Validation\n');
fprintf('==========================================================\n');

% 1. Generate RF Signal & Parameters
fprintf('[1/5] Simulating RF signal in MATLAB...\n');
[rfData, params] = simulateRFData();
[numSamples, numElements] = size(rfData);
Nz = 224; Nx = 224;

% 2. Run MATLAB CPU Reconstruction Benchmark (50 iterations)
fprintf('[2/5] Running MATLAB CPU Reconstruction Benchmark...\n');
% Warmup
[bModeMATLAB, envMATLAB, ~] = reconstructDAS(rfData, params, [Nz, Nx], 50);

numIters = 20;
tic;
for k = 1:numIters
    [~, ~, ~] = reconstructDAS(rfData, params, [Nz, Nx], 50);
end
tMATLAB_total = toc;
tMATLAB_ms = (tMATLAB_total / numIters) * 1000;
fprintf('  MATLAB CPU Latency : %.3f ms / frame (%.2f FPS)\n', tMATLAB_ms, 1000/tMATLAB_ms);

% 3. Export Data to Binary Files for CUDA
fprintf('\n[3/5] Exporting binary data for CUDA program...\n');
cudaDir = fullfile(repoRoot, 'reconstruction', 'cuda');
rfBinPath = fullfile(cudaDir, 'rf_data.bin');
paramsBinPath = fullfile(cudaDir, 'params.bin');
bmodeCudaPath = fullfile(cudaDir, 'bmode_cuda.bin');
envCudaPath = fullfile(cudaDir, 'envelope_cuda.bin');

% Write RF Data (row-major single float)
fid = fopen(rfBinPath, 'wb');
fwrite(fid, single(rfData'), 'float32'); % transpose so row-major in C++
fclose(fid);

% Write DASParams struct
% Struct layout: c(float), fs(float), f0(float), numElements(int32), numSamples(int32),
% Nz(int32), Nx(int32), zMin(float), zMax(float), xMin(float), xMax(float), dynamicRangedB(float)
zMin = 5.0e-3; zMax = 45.0e-3;
xMin = min(params.elementPositionsX);
xMax = max(params.elementPositionsX);

fid = fopen(paramsBinPath, 'wb');
fwrite(fid, single(params.c), 'float32');
fwrite(fid, single(params.fs), 'float32');
fwrite(fid, single(params.f0), 'float32');
fwrite(fid, int32(params.numElements), 'int32');
fwrite(fid, int32(numSamples), 'int32');
fwrite(fid, int32(Nz), 'int32');
fwrite(fid, int32(Nx), 'int32');
fwrite(fid, single(zMin), 'float32');
fwrite(fid, single(zMax), 'float32');
fwrite(fid, single(xMin), 'float32');
fwrite(fid, single(xMax), 'float32');
fwrite(fid, single(50.0), 'float32');
fclose(fid);

% 4. Execute CUDA Program in WSL
fprintf('\n[4/5] Executing CUDA DAS Executable in WSL...\n');
cmdStr = sprintf('wsl -d Ubuntu-22.04 bash -c "/mnt/c/Users/anura/Desktop/RealTimeMedicalImaging/reconstruction/cuda/main_das_cuda /mnt/c/Users/anura/Desktop/RealTimeMedicalImaging/reconstruction/cuda/rf_data.bin /mnt/c/Users/anura/Desktop/RealTimeMedicalImaging/reconstruction/cuda/params.bin /mnt/c/Users/anura/Desktop/RealTimeMedicalImaging/reconstruction/cuda/bmode_cuda.bin /mnt/c/Users/anura/Desktop/RealTimeMedicalImaging/reconstruction/cuda/envelope_cuda.bin"');
[status, cmdOut] = system(cmdStr);
disp(cmdOut);
if status ~= 0
    error('CUDA execution failed with status %d', status);
end

% 5. Import CUDA Results & Compare Numerical Parity
fprintf('[5/5] Comparing Numerical Parity & Spatial Accuracy...\n');
fid = fopen(bmodeCudaPath, 'rb');
bModeCUDA_raw = fread(fid, [3, Nz * Nx], 'uint8=>uint8')';
fclose(fid);

% C++ d_bmode is row-major (z * Nx + x) x 3 channels.
% Reshape([Nx, Nz])' transposes row-major C++ rows to MATLAB column-major layout.
grayCUDA_2D = reshape(bModeCUDA_raw(:, 1), [Nx, Nz])';
bModeCUDA = cat(3, grayCUDA_2D, grayCUDA_2D, grayCUDA_2D);

fid = fopen(envCudaPath, 'rb');
envCUDA = reshape(fread(fid, [Nx, Nz], 'float32=>double'), [Nx, Nz])';
fclose(fid);

% Integrity Checks
hasNaN = any(isnan(bModeCUDA(:)));
hasInf = any(isinf(bModeCUDA(:)));

diffImg = abs(double(bModeMATLAB) - double(bModeCUDA));
maxAbsDiff = max(diffImg(:));
meanAbsDiff = mean(diffImg(:));

% Peak Signal-to-Noise Ratio (PSNR)
mse = mean((double(bModeMATLAB(:)) - double(bModeCUDA(:))).^2);
if mse == 0
    psnrVal = Inf;
else
    psnrVal = 10 * log10(255^2 / mse);
end

% Normalized Cross-Correlation (NCC)
matFlat = double(bModeMATLAB(:)) - mean(double(bModeMATLAB(:)));
cudaFlat = double(bModeCUDA(:)) - mean(double(bModeCUDA(:)));
nccVal = sum(matFlat .* cudaFlat) / (sqrt(sum(matFlat.^2)) * sqrt(sum(cudaFlat.^2)));

fprintf('=== Image Parity Metrics (MATLAB CPU vs CUDA GPU) ===\n');
fprintf('  Finite Check (No NaN/Inf) : %s\n', select(~hasNaN && ~hasInf, "PASSED", "FAILED"));
fprintf('  Max Absolute Pixel Diff   : %.2f / 255\n', maxAbsDiff);
fprintf('  Mean Absolute Pixel Diff  : %.4f / 255\n', meanAbsDiff);
fprintf('  PSNR                      : %.2f dB\n', psnrVal);
fprintf('  Normalized Cross-Corr     : %.6f\n', nccVal);

% 2D Spatial Scatterer Target Comparison (5 Targets)
depthsMM = linspace(zMin*1e3, zMax*1e3, Nz);
lateralsMM = linspace(xMin*1e3, xMax*1e3, Nx);

targets = params.scatterers(1:5, :); % First 5 hyperechoic point targets
fprintf('\n=== 2D Spatial Target Peak Verification (5 Scatterer Targets) ===\n');

for k = 1:size(targets, 1)
    expX = targets(k, 1) * 1e3;
    expZ = targets(k, 2) * 1e3;
    
    [~, rIdx] = min(abs(depthsMM - expZ));
    [~, cIdx] = min(abs(lateralsMM - expX));
    
    rWin = max(1, rIdx-10) : min(Nz, rIdx+10);
    cWin = max(1, cIdx-10) : min(Nx, cIdx+10);
    
    subEnvCUDA = envCUDA(rWin, cWin);
    [~, maxIdx] = max(subEnvCUDA(:));
    [localR, localC] = ind2sub(size(subEnvCUDA), maxIdx);
    
    foundX = lateralsMM(cWin(localC));
    foundZ = depthsMM(rWin(localR));
    distErr = sqrt((foundX - expX)^2 + (foundZ - expZ)^2);
    
    fprintf('  Target %d [Simulated: x=%+5.1f mm, z=%4.1f mm] -> CUDA Peak: [x=%+5.1f mm, z=%4.1f mm] (Dist Error: %.2f mm)\n', ...
        k, expX, expZ, foundX, foundZ, distErr);
end

% Save CUDA B-mode preview to docs/
cudaPreviewPath = fullfile(repoRoot, 'docs', 'cuda_rf_bmode_preview.png');
imwrite(bModeCUDA, cudaPreviewPath);
fprintf('\nCUDA B-mode Preview Saved: %s\n', cudaPreviewPath);
fprintf('==========================================================\n');

function out = select(cond, valTrue, valFalse)
    if cond
        out = valTrue;
    else
        out = valFalse;
    end
end
