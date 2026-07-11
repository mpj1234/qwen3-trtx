cmake_minimum_required(VERSION 4.2)
project(qwen3_trtx LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /utf-8")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8")

set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

if(MSVC)
    file(GLOB _msvc_versions LIST_DIRECTORIES true "D:/VS2022/Professional/VC/Tools/MSVC/*")
    list(SORT _msvc_versions COMPARE NATURAL ORDER DESCENDING)
    list(GET _msvc_versions 0 MSVC_TOOLS_DIR)

    file(GLOB _winsdk_versions LIST_DIRECTORIES true "C:/Program Files (x86)/Windows Kits/10/Include/*")
    list(SORT _winsdk_versions COMPARE NATURAL ORDER DESCENDING)
    list(GET _winsdk_versions 0 WINDOWS_SDK_INCLUDE_DIR)

    file(GLOB _winsdk_lib_versions LIST_DIRECTORIES true "C:/Program Files (x86)/Windows Kits/10/Lib/*")
    list(SORT _winsdk_lib_versions COMPARE NATURAL ORDER DESCENDING)
    list(GET _winsdk_lib_versions 0 WINDOWS_SDK_LIB_DIR)

    set(MSVC_STANDARD_INCLUDE_DIRS
            "${MSVC_TOOLS_DIR}/include"
            "${WINDOWS_SDK_INCLUDE_DIR}/ucrt"
            "${WINDOWS_SDK_INCLUDE_DIR}/shared"
            "${WINDOWS_SDK_INCLUDE_DIR}/um"
            "${WINDOWS_SDK_INCLUDE_DIR}/winrt"
            "${WINDOWS_SDK_INCLUDE_DIR}/cppwinrt")

    include_directories(SYSTEM ${MSVC_STANDARD_INCLUDE_DIRS})

    set(MSVC_STANDARD_LIBRARY_DIRS
            "${MSVC_TOOLS_DIR}/lib/x64"
            "${WINDOWS_SDK_LIB_DIR}/ucrt/x64"
            "${WINDOWS_SDK_LIB_DIR}/um/x64")
endif()

set(OpenCV_DIR E:\\Opencv\\install\\opencv-4.8.0\\build\\x64\\vc16\\lib)
find_package(OpenCV REQUIRED)

set(CUDA_TOOLKIT_ROOT_DIR C:\\Program\ Files\\NVIDIA\ GPU\ Computing\ Toolkit\\CUDA\\v12.9)
set(TENSORRT_ROOT E:\\TensorRT\\TensorRT-10.10.0.31)
# set(TENSORRT_ROOT E:\\TensorRT\\TensorRT-10.16.1.11)

add_subdirectory(${PROJECT_SOURCE_DIR}/tokenizer.cpp)

aux_source_directory(${CMAKE_SOURCE_DIR}/src SRC_FILES)
file(GLOB PLUGIN_FILES
        ${CMAKE_SOURCE_DIR}/plugin/*.cpp
        ${CMAKE_SOURCE_DIR}/plugin/*.cu)

add_executable(qwen3_trtx main.cpp ${SRC_FILES} ${PLUGIN_FILES})
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CXX)
foreach(src_file IN LISTS SRC_FILES)
    set_source_files_properties(${src_file} PROPERTIES LANGUAGE CXX)
endforeach()

target_include_directories(qwen3_trtx PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_TOOLKIT_ROOT_DIR}/include
        ${TENSORRT_ROOT}/include
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/plugin
        ${CMAKE_SOURCE_DIR}/src)

target_compile_definitions(qwen3_trtx PRIVATE
        NOMINMAX
        API_EXPORTS
        UTF8PROC_EXPORTS)

target_link_directories(qwen3_trtx PRIVATE
        ${OpenCV_LIB_DIR}
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
        ${TENSORRT_ROOT}/lib
        ${MSVC_STANDARD_LIBRARY_DIRS}
        ${CMAKE_SOURCE_DIR}/lib)

target_link_libraries(qwen3_trtx PRIVATE
        tokenizer_lib
        opencv_core
        opencv_highgui
        opencv_imgproc
        opencv_imgcodecs
        cudart
        cublas
        nvinfer_10
        onig)
