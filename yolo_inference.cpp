#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include <opencv2/opencv.hpp>

// Logger for TensorRT
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
} logger;

// Calculate size of a binding
size_t getSizeByDim(const nvinfer1::Dims& dims) {
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        size *= dims.d[i];
    }
    return size;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <engine_path> <image_path>" << std::endl;
        return -1;
    }

    std::string enginePath = argv[1];
    std::string imagePath = argv[2];

    // 1. Load Engine
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
    if (!file.good()) {
        std::cerr << "Error reading engine file" << std::endl;
        return -1;
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "Error reading engine file content" << std::endl;
        return -1;
    }

    // 2. Create Runtime and Engine
    std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    std::unique_ptr<nvinfer1::ICudaEngine> engine{runtime->deserializeCudaEngine(buffer.data(), size)};
    std::unique_ptr<nvinfer1::IExecutionContext> context{engine->createExecutionContext()};

    // 3. Prepare Inputs/Outputs
    // Assuming YOLOv11 export (checkout ultralytics export log for exact binding names)
    // Usually 'images' for input and 'output0' for output
    int32_t inputIndex = engine->getBindingIndex("images");
    int32_t outputIndex = engine->getBindingIndex("output0");

    if (inputIndex == -1 || outputIndex == -1) {
        std::cerr << "Could not find binding indices. Check your export!" << std::endl;
        // fallback to 0 and 1
        inputIndex = 0;
        outputIndex = 1;
    }

    void* buffers[2];
    nvinfer1::Dims inputDims = engine->getBindingDimensions(inputIndex);
    nvinfer1::Dims outputDims = engine->getBindingDimensions(outputIndex);

    // Dynamic shapes might have -1, assuming fixed size for demo (e.g., 640x640)
    // For production, handle dynamic shapes properly using context->setBindingDimensions
    int inputH = inputDims.d[2] > 0 ? inputDims.d[2] : 640;
    int inputW = inputDims.d[3] > 0 ? inputDims.d[3] : 640;
    
    // Explicitly set dimensions if dynamic
    context->setBindingDimensions(inputIndex, nvinfer1::Dims4(1, 3, inputH, inputW));
    
    size_t inputSize = 1 * 3 * inputH * inputW * sizeof(float);
    // Be careful with output size calculation if dimensions are dynamic (-1)
    size_t outputSize = 1 * outputDims.d[1] * outputDims.d[2] * sizeof(float); // typically [1, 84, 8400] for YOLOv8/11

    cudaMalloc(&buffers[inputIndex], inputSize);
    cudaMalloc(&buffers[outputIndex], outputSize);

    // 4. Preprocess Image (OpenCV)
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        std::cerr << "Image not found" << std::endl;
        return -1;
    }

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(inputW, inputH));
    
    // HWC to CHW and Normalize
    cv::Mat floatImg;
    resized.convertTo(floatImg, CV_32F, 1.0/255.0); // 0-1 normalization

    // Split channels
    std::vector<cv::Mat> channels(3);
    cv::split(floatImg, channels);
    
    std::vector<float> inputData;
    for(auto& channel : channels) {
        std::vector<float> data;
        data.assign((float*)channel.datastart, (float*)channel.dataend);
        inputData.insert(inputData.end(), data.begin(), data.end());
    }

    // 5. Inference
    cudaMemcpy(buffers[inputIndex], inputData.data(), inputSize, cudaMemcpyHostToDevice);
    
    // Using enqueueV2 for newer TensorRT
    context->enqueueV2(buffers, 0, nullptr);

    std::vector<float> outputData(outputSize / sizeof(float));
    cudaMemcpy(outputData.data(), buffers[outputIndex], outputSize, cudaMemcpyDeviceToHost);

    // 6. Post-process
    // outputData contains [1, 4+cls, detections] typically
    // You need to parse xywh, confidence, class, apply NMS, etc.
    // This is complex for pure C++ from scratch.
    
    std::cout << "Inference successful!" << std::endl;
    std::cout << "Output shape: [" << 1 << ", " << outputDims.d[1] << ", " << outputDims.d[2] << "]" << std::endl;
    std::cout << "First few output values: " << outputData[0] << ", " << outputData[1] << ", " << outputData[2] << std::endl;

    // Cleanup
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputIndex]);

    return 0;
}
