function scores = resnetPredict(inputImage) %#codegen
%RESNETPREDICT CUDA code-generation entry point for ResNet-18.
%
% Input:
%   inputImage - 224x224x3 ultrasound image
%
% Output:
%   scores - 3-class prediction scores

    persistent net

    if isempty(net)
        net = coder.loadDeepLearningNetwork("netResNet.mat");
    end

    scores = predict(net, inputImage);

end