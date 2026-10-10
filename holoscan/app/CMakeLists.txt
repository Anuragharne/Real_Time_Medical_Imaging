cmake_minimum_required(VERSION 3.20)

project(resnet18_holoscan LANGUAGES CXX CUDA)

find_package(holoscan 4.6 REQUIRED CONFIG
    PATHS "/opt/nvidia/holoscan"
)

find_package(CUDAToolkit REQUIRED)

find_package(OpenCV REQUIRED COMPONENTS
    core
    imgcodecs
    imgproc
)

add_executable(resnet18_holoscan
    main.cpp
    ../../reconstruction/cuda/das_kernel.cu
)

target_include_directories(resnet18_holoscan
    PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../reconstruction/cuda
)

target_link_libraries(resnet18_holoscan
    PRIVATE
    holoscan::core
    holoscan::ops::inference
    holoscan::ops::ping_tensor_tx
    CUDA::cudart
    CUDA::cufft
    ${OpenCV_LIBS}
)

