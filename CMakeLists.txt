cmake_minimum_required(VERSION 3.10)
project(YOLOv11_TensorRT)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find CUDA
find_package(CUDA REQUIRED)

# Find OpenCV (for image processing)
find_package(OpenCV REQUIRED)

# Set Path to TensorRT (modify if not in standard path)
# JetPack usually installs TensorRT in /usr/lib/aarch64-linux-gnu/ and /usr/include/aarch64-linux-gnu/
# If compiling on PC, set specific paths
if(EXISTS "/usr/src/tensorrt")
    include_directories("/usr/src/tensorrt/include")
    link_directories("/usr/src/tensorrt/lib")
elseif(EXISTS "/usr/include/aarch64-linux-gnu")
    # Typical Jetson path
    include_directories("/usr/include/aarch64-linux-gnu")
    link_directories("/usr/lib/aarch64-linux-gnu")
endif()

# Executable
add_executable(yolo_cpp yolo_inference.cpp)

# Link Libraries
target_link_libraries(yolo_cpp
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    nvinfer
    nvparsers
    nvonnxparser
    cudart
)

# Compiler warnings
if(MSVC)
    target_compile_options(yolo_cpp PRIVATE /W4)
else()
    target_compile_options(yolo_cpp PRIVATE -Wall -Wextra)
endif()
