cmake_minimum_required(VERSION 4.2)
project(qwen3_trtx)

set(CMAKE_CXX_STANDARD 17)
# 设置nvcc编译cu文件时候使用utf-8编码
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /utf-8")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")

enable_language(CUDA)

# 设置cuda多个框架支持
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

# OpenCV
set(OpenCV_DIR E:\\Opencv\\install\\opencv-4.8.0\\build\\x64\\vc16\\lib)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

# CUDA
set(CUDA_TOOLKIT_ROOT_DIR C:\\Program\ Files\\NVIDIA\ GPU\ Computing\ Toolkit\\CUDA\\v12.9)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)

# TensorRT
set(TENSORRT_ROOT E:\\TensorRT\\TensorRT-10.10.0.31)
#set(TENSORRT_ROOT E:\\TensorRT\\TensorRT-10.16.1.11)
include_directories(${TENSORRT_ROOT}/include)
link_directories(${TENSORRT_ROOT}/lib)

link_libraries(
        opencv_core
        opencv_highgui
        opencv_imgproc
        opencv_imgcodecs
        cudart
        cublas
        nvinfer_10
)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/plugin)
include_directories(${CMAKE_SOURCE_DIR}/src)
link_directories(${CMAKE_SOURCE_DIR}/lib)

add_definitions(-DNOMINMAX)

add_definitions(-DAPI_EXPORTS)

include_directories(${CMAKE_SOURCE_DIR}/include)
aux_source_directory(${CMAKE_SOURCE_DIR}/src SRC_FILES)
file(GLOB PLUGIN_FILES
        ${CMAKE_SOURCE_DIR}/plugin/*.cpp
        ${CMAKE_SOURCE_DIR}/plugin/*.cu)

add_executable(qwen3_trtx main.cpp ${SRC_FILES} ${PLUGIN_FILES})
