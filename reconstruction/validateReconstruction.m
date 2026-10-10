% Validation & Benchmarking Script for RF-to-Ultrasound DAS Reconstruction
repoRoot = 'c:\Users\anura\Desktop\RealTimeMedicalImaging';
addpath(fullfile(repoRoot, 'reconstruction'));
addpath(fullfile(repoRoot, 'preprocessing'));
addpath(fullfile(repoRoot, 'models'));

fprintf('==========================================================\n');
fprintf(' RF-to-Ultrasound DAS Reconstruction Prototype\n');
fprintf('==========================================================\n');

% 1. Simulate Physically Meaningful Multichannel RF Data
fprintf('[1/4] Simulating ultrasound RF array signals...\n');
[rfData, params] = simulateRFData();
fprintf('  Array Elements   : %d\n', params.numElements);
fprintf('  RF Signal Shape  : %d samples x %d channels\n', size(rfData, 1), size(rfData, 2));
fprintf('  Sampling Freq    : %.1f MHz\n', params.fs / 1e6);
fprintf('  Sound Speed (c)  : %d m/s\n', params.c);

% 2. Perform DAS Beamforming & B-mode Image Reconstruction
fprintf('\n[2/4] Running Delay-and-Sum (DAS) Beamforming & B-mode Formation...\n');
[bModeImage, envelopeImg, timingMs] = reconstructDAS(rfData, params, [224, 224], 50);

fprintf('  Reconstruction Latency : %.3f ms\n', timingMs);
fprintf('  Output Image Size      : %d x %d x %d\n', size(bModeImage, 1), size(bModeImage, 2), size(bModeImage, 3));
fprintf('  Output Class           : %s\n', class(bModeImage));

% 3. Numerical Integrity Checks
fprintf('\n[3/4] Validating Image Numerical Integrity...\n');
hasNaN = any(isnan(bModeImage(:)));
hasInf = any(isinf(bModeImage(:)));
minVal = min(bModeImage(:));
maxVal = max(bModeImage(:));

fprintf('  Finite Check (No NaN/Inf) : %s\n', select(~hasNaN && ~hasInf, "PASSED", "FAILED"));
fprintf('  Pixel Value Range         : Min = %d, Max = %d (Expected: 0 to 255)\n', minVal, maxVal);

% Save preview image to docs/
docsDir = fullfile(repoRoot, 'docs');
if ~exist(docsDir, 'dir')
    mkdir(docsDir);
end
previewPath = fullfile(docsDir, 'rf_bmode_preview.png');
imwrite(bModeImage, previewPath);
fprintf('  B-mode Preview Saved      : %s\n', previewPath);

% 4. Exploratory Inference on ResNet-18 Classifier (Out-of-Domain Labelled)
fprintf('\n[4/4] Exploratory ResNet-18 Classifier Inference...\n');
modelPath = fullfile(repoRoot, 'models', 'netResNet.mat');
if exist(modelPath, 'file')
    loaded = load(modelPath);
    if isfield(loaded, 'netResNet')
        net = loaded.netResNet;
    elseif isfield(loaded, 'net')
        net = loaded.net;
    else
        fns = fieldnames(loaded);
        net = loaded.(fns{1});
    end
    classNames = net.Layers(end).Classes;
    
    scores = predict(net, bModeImage);
    [~, idx] = max(scores);
    predClass = string(classNames(idx));
    
    fprintf('  Exploratory Prediction : %s\n', predClass);
    fprintf('  Exploratory Scores     : Benign=%.6f, Malignant=%.6f, Normal=%.6f\n', scores(1), scores(2), scores(3));
    fprintf('  *NOTE*: ResNet-18 was fine-tuned on BUSI B-mode images. Simulated phantom RF reconstruction is out-of-domain.\n');
end

fprintf('\n==========================================================\n');
fprintf(' Reconstruction Prototype Completed Successfully!\n');
fprintf('==========================================================\n');

function out = select(cond, valTrue, valFalse)
    if cond
        out = valTrue;
    else
        out = valFalse;
    end
end
