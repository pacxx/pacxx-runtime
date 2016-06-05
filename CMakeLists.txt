cmake_minimum_required(VERSION 3.0)
project(PACXX_V2)

find_package(LLVM REQUIRED CONFIG)
find_package(CUDA REQUIRED)

include_directories(${CUDA_TOOLKIT_INCLUDE})

add_compile_options(-std=c++14 -stdlib=libc++)

add_definitions(${LLVM_DEFINITIONS})

set(PACXX_DEV_FLAGS -std=pacxx -O0 -c -emit-llvm -target pacxx-unknown-unknown -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -D__CUDA_DEVICE_CODE -isystem $ENV{PACXX_PATH}/include/c++/v1 -isystem $ENV{PACXX_PATH}include -isystem ${CUDA_TOOLKIT_INCLUDE} -I ${CMAKE_SOURCE_DIR}/include)

include_directories("./include")

add_library(pacxxrt2 ${SOURCE_FILES} include/Executor.h include/detail/IRCompiler.h src/PTXBackend.cpp include/detail/cuda/PTXBackend.h include/detail/common/Exceptions.h include/detail/common/Log.h include/detail/common/Common.h src/Common.cpp include/detail/CoreInitializer.h src/CoreInitializer.cpp include/ModuleLoader.h src/ModuleLoader.cpp include/detail/IRRuntime.h include/detail/cuda/CUDARuntime.h src/CUDARuntime.cpp include/detail/cuda/CUDAErrorDetection.h src/CUDAErrorDetection.cpp include/CodePolicy.h include/detail/KernelConfiguration.h src/KernelConfiguration.cpp include/detail/Kernel.h include/detail/cuda/CUDAKernel.h src/CUDAKernel.cpp include/detail/DeviceBuffer.h include/detail/cuda/CUDADeviceBuffer.h include/detail/MemoryManager.h include/detail/common/Meta.h include/detail/stl_extension/Allocators.h.h src/Log.cpp src/Executor.cpp)

add_custom_command(TARGET pacxxrt2 POST_BUILD COMMAND $ENV{PACXX_PATH}/bin/llvm-as ${CMAKE_SOURCE_DIR}/src/bitcode/syscalls.ll -o ${CMAKE_BINARY_DIR}/syscalls.bc)
add_custom_command(TARGET pacxxrt2 POST_BUILD COMMAND $ENV{PACXX_PATH}/bin/clang ${PACXX_DEV_FLAGS} ${CMAKE_SOURCE_DIR}/src/bitcode/nvvm_device_binding.cpp -o ${CMAKE_BINARY_DIR}/nvvm_device_binding.bc)
add_custom_command(TARGET pacxxrt2 POST_BUILD COMMAND $ENV{PACXX_PATH}/bin/clang ${PACXX_DEV_FLAGS} ${CMAKE_SOURCE_DIR}/src/DeviceCode.cpp -o ${CMAKE_BINARY_DIR}/DeviceCode.bc)