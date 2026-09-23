function outputImage = preprocessPadding(filename,padValue)
%PREPROCESSPADDING Preserve aspect ratio and use fixed padding intensity.

    targetSize = [224 224];

    % Read image
    inputImage = imread(filename);

    % Convert RGB to grayscale
    if size(inputImage,3) == 3
        inputImage = rgb2gray(inputImage);
    end

    % Convert to single precision
    inputImage = im2single(inputImage);

    % Median filtering
    inputImage = medfilt2(inputImage,[3 3]);

    % Adaptive histogram equalization
    inputImage = adapthisteq(inputImage);

    % Preserve aspect ratio
    scale = min(targetSize ./ size(inputImage));

    newSize = round(size(inputImage) * scale);

    resizedImage = imresize(inputImage,newSize);

    % Calculate required padding
    padRows = targetSize(1) - size(resizedImage,1);
    padCols = targetSize(2) - size(resizedImage,2);

    top = floor(padRows/2);
    bottom = padRows - top;

    left = floor(padCols/2);
    right = padCols - left;

    % Fixed padding value for ALL images
    outputImage = padarray( ...
        resizedImage,[top left],padValue,"pre");

    outputImage = padarray( ...
        outputImage,[bottom right],padValue,"post");

end