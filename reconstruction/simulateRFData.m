function [rfData, params] = simulateRFData()
% SIMULATERFDATA Simulates physically meaningful raw RF ultrasound array signals.
%
% Generates multichannel RF data for a 64-element linear array imaging a phantom
% containing hyperechoic point targets and diffuse background speckle scatterers.
%
% Output:
%   rfData - (numSamples x numElements) matrix of raw RF signals
%   params - struct containing acoustic and array parameters

    % 1. Acoustic & Array Geometry Parameters
    params.c = 1540;                 % Speed of sound in tissue (m/s)
    params.f0 = 5.0e6;               % Transducer center frequency (Hz)
    params.fs = 40.0e6;              % Sampling frequency (Hz)
    params.numElements = 64;         % Number of array elements
    params.pitch = params.c / params.f0; % Element pitch (m) = lambda = 0.308 mm
    
    % Array element lateral positions x_i (m) centered at 0
    elementX = ((1:params.numElements) - (params.numElements + 1) / 2) * params.pitch;
    params.elementPositionsX = elementX;
    
    % Depth range
    zMin = 5.0e-3;                   % 5 mm depth
    zMax = 45.0e-3;                  % 45 mm depth
    tMax = (zMax * 2) / params.c;    % Max propagation time
    t = (0 : (1/params.fs) : tMax)'; % Time vector
    params.timeVector = t;
    numSamples = length(t);
    
    % 2. Phantom Scatterers (x, z, amplitude)
    % Define hyperechoic point targets (simulated lesions/micro-calcifications)
    scatterers = [
        0.000,  0.015, 1.0;  % Target 1: center, depth 15 mm
       -0.005,  0.025, 0.9;  % Target 2: left -5mm, depth 25 mm
        0.005,  0.025, 0.9;  % Target 3: right +5mm, depth 25 mm
        0.000,  0.035, 0.8;  % Target 4: center, depth 35 mm
       -0.008,  0.035, 0.7;  % Target 5: left -8mm, depth 35 mm
    ];
    
    % Add diffuse background speckle scatterers
    rng(42); % Fixed seed for reproducibility
    numSpeckle = 150;
    speckleX = (rand(numSpeckle, 1) - 0.5) * (params.numElements * params.pitch);
    speckleZ = zMin + rand(numSpeckle, 1) * (zMax - zMin);
    speckleAmp = (rand(numSpeckle, 1) * 0.15);
    
    allScatterers = [scatterers; [speckleX, speckleZ, speckleAmp]];
    params.scatterers = allScatterers;
    
    % 3. RF Signal Synthesis
    % Transducer pulse impulse response: Gaussian-modulated sinusoid
    pulseWidth = 2.0 / params.f0;
    tPulse = (-pulseWidth : (1/params.fs) : pulseWidth)';
    pulse = exp(-(tPulse / (pulseWidth / 2.5)).^2) .* cos(2 * pi * params.f0 * tPulse);
    
    rfData = zeros(numSamples, params.numElements);
    
    % Calculate round-trip delay from plane-wave tx (normal) to each element rx
    for elem = 1:params.numElements
        rxX = elementX(elem);
        elemSignal = zeros(numSamples, 1);
        
        for k = 1:size(allScatterers, 1)
            scatX = allScatterers(k, 1);
            scatZ = allScatterers(k, 2);
            scatAmp = allScatterers(k, 3);
            
            % Round-trip time: Plane wave tx (depth z) + spherical wave rx to element at rxX
            tau = (scatZ + sqrt((scatX - rxX)^2 + scatZ^2)) / params.c;
            
            % Sample index for delay
            sampleIdx = round(tau * params.fs) + 1;
            if sampleIdx > 0 && (sampleIdx + length(pulse) - 1) <= numSamples
                range = sampleIdx : (sampleIdx + length(pulse) - 1);
                elemSignal(range) = elemSignal(range) + scatAmp * pulse;
            end
        end
        rfData(:, elem) = elemSignal;
    end
end
