cmake_minimum_required(VERSION 2.8)
project(PACXX_V2)

find_package(LLVM REQUIRED CONFIG)
find_package(CUDA REQUIRED)

include_directories(${CUDA_TOOLKIT_INCLUDE})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

add_definitions(${LLVM_DEFINITIONS})

include_directories("./include")

add_library(pacxxrt2 ${SOURCE_FILES} include/Executor.h include/detail/IRCompiler.h src/PTXBackend.cpp include/detail/cuda/PTXBackend.h include/detail/common/Exceptions.h include/detail/common/Log.h include/detail/common/Common.h src/Common.cpp include/detail/CoreInitializer.h src/CoreInitializer.cpp include/ModuleLoader.h src/ModuleLoader.cpp include/detail/IRRuntime.h include/detail/cuda/CUDARuntime.h src/CUDARuntime.cpp include/detail/cuda/CUDAErrorDetection.h src/CUDAErrorDetection.cpp include/CodePolicy.h include/detail/KernelConfiguration.h src/KernelConfiguration.cpp include/detail/Kernel.h include/detail/cuda/CUDAKernel.h src/CUDAKernel.cpp include/detail/DeviceBuffer.h include/detail/cuda/CUDADeviceBuffer.h)