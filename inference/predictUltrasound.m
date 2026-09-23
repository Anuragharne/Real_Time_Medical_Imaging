function [predictedLabel, scores, processedImage] = predictUltrasound(filename)
%PREDICTULTRASOUND Classify a breast ultrasound image using ResNet-18.
%
% Input:
%   filename - path to ultrasound image
%
% Outputs:
%   predictedLabel - predicted class
%   scores         - classification scores
%   processedImage - preprocessed 224x224x3 image

    % Load trained ResNet-18 model once
    persistent net

    if isempty(net)
        modelData = load( ...
            "C:\Users\anura\Desktop\RealTimeMedicalImaging\models\netResNet.mat", ...
            "netResNet");

        net = modelData.netResNet;
    end

    % Preprocess image using the SAME preprocessing
    % used during ResNet-18 training
    processedImage = preprocessResNet(filename);

    % Classification using GPU
    [predictedLabel, scores] = classify( ...
        net, ...
        processedImage, ...
        ExecutionEnvironment="gpu");

end