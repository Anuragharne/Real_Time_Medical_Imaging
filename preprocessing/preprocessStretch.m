function outputImage = preprocessStretch(filename)
%PREPROCESSSTRETCH Preprocess and directly resize a BUSI image.
%
% Input:
%   filename - path to the ultrasound image
%
% Output:
%   outputImage - 224 x 224 grayscale image

    % Read image from filename
    inputImage = imread(filename);

    % Convert RGB image to grayscale
    if size(inputImage,3) == 3
        inputImage = rgb2gray(inputImage);
    end

    % Convert to single precision and normalize to [0,1]
    inputImage = im2single(inputImage);

    % Median filtering
    inputImage = medfilt2(inputImage,[3 3]);

    % Adaptive histogram equalization (CLAHE)
    inputImage = adapthisteq(inputImage);

    % Direct resize - aspect ratio is not preserved
    outputImage = imresize(inputImage,[224 224]);

end