function [predictedLabel, confidence, scores, inferenceTime] = ...
    predictUltrasoundResNet(filename, net)

% PREDICTULTRASOUNDRESNET
% Performs breast ultrasound classification using trained ResNet-18.
%
% Inputs:
%   filename - path to ultrasound image
%   net      - trained ResNet-18 network
%
% Outputs:
%   predictedLabel - predicted class
%   confidence     - confidence of prediction
%   scores         - probability for each class
%   inferenceTime  - inference time in seconds

    % ---------------------------------------------------------
    % 1. Preprocess the ultrasound image
    % ---------------------------------------------------------
    processedImage = preprocessResNet(filename);

    % ---------------------------------------------------------
    % 2. ResNet-18 inference
    % ---------------------------------------------------------
    tic;

    [predictedLabel, scores] = classify( ...
        net, ...
        processedImage, ...
        ExecutionEnvironment="gpu");

    inferenceTime = toc;

    % ---------------------------------------------------------
    % 3. Extract prediction confidence
    % ---------------------------------------------------------
    confidence = max(scores);

end