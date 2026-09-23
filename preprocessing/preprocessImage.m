function outputImage = preprocessImage(inputImage)
%PREPROCESSIMAGE Preprocess a BUSI ultrasound image.
%
%   outputImage = preprocessImage(inputImage)
%
%   Pipeline:
%       1. Convert to grayscale
%       2. Convert to single precision
%       3. Median filtering
%       4. Adaptive contrast enhancement

    % Convert RGB image to grayscale when necessary
    if size(inputImage,3) == 3
        inputImage = rgb2gray(inputImage);
    end

    % Convert to single precision and normalize to [0,1]
    inputImage = im2single(inputImage);

    % Median filtering
    filteredImage = medfilt2(inputImage,[3 3]);

    % Adaptive histogram equalization (CLAHE)
    outputImage = adapthisteq(filteredImage);

end