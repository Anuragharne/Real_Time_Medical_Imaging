function outputImage = preprocessResNet(filename)

    inputImage = imread(filename);

    % Convert RGB image to grayscale
    if size(inputImage,3) == 3
        inputImage = rgb2gray(inputImage);
    end

    % Resize to ResNet input size
    inputImage = imresize(inputImage,[224 224]);

    % Convert grayscale to 3 identical channels
    outputImage = cat(3,inputImage,inputImage,inputImage);

end