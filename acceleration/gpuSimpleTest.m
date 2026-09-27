function y = gpuSimpleTest(x) %#codegen

coder.gpu.kernelfun();

y = x .* 2;

end