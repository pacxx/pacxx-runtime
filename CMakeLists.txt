cmake_minimum_required(VERSION 3.0)
project(PACXX_V2)

set(TBB_INCLUDE_DIR "$ENV{PACXX_PATH}/include" CACHE FILEPATH "include directory of tbb")

set(LLVM_DIR $ENV{PACXX_PATH}/lib/cmake/llvm/)

find_package(LLVM REQUIRED CONFIG)
find_package(CUDA REQUIRED)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CUDA_TOOLKIT_INCLUDE})
include_directories(${TBB_INCLUDE_DIR})

set(NVPTX_DIR "" CACHE PATH "Path to the NVPTX backend")

include_directories(${NVPTX_DIR})

add_compile_options(-std=c++14)

add_definitions(${LLVM_DEFINITIONS})

set(PACXX_DEV_FLAGS -std=pacxx -O0 -c -emit-llvm -target nvptx64-unknown-unknown -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -D__CUDA_DEVICE_CODE -isystem $ENV{PACXX_PATH}/include/c++/v1 -isystem $ENV{PACXX_PATH}include -isystem ${CUDA_TOOLKIT_INCLUDE} -I ${CMAKE_SOURCE_DIR}/include)

include_directories("./include")

add_library(pacxxrt2 ${SOURCE_FILES} include/Executor.h src/cuda/PTXBackend.cpp include/detail/cuda/PTXBackend.h include/detail/common/Exceptions.h include/detail/common/Log.h include/detail/common/Common.h src/Common.cpp include/detail/CoreInitializer.h src/CoreInitializer.cpp include/ModuleLoader.h src/ModuleLoader.cpp include/detail/IRRuntime.h include/detail/cuda/CUDARuntime.h src/cuda/CUDARuntime.cpp include/detail/cuda/CUDAErrorDetection.h src/cuda/CUDAErrorDetection.cpp include/CodePolicy.h include/detail/KernelConfiguration.h src/KernelConfiguration.cpp include/detail/Kernel.h include/detail/cuda/CUDAKernel.h src/cuda/CUDAKernel.cpp include/detail/DeviceBuffer.h include/detail/cuda/CUDADeviceBuffer.h include/detail/MemoryManager.h include/detail/common/Meta.h include/detail/stl_extension/Allocators.h.h src/Log.cpp src/Executor.cpp include/detail/common/Timing.h include/detail/cuda/CUDAUtility.h include/detail/native/NativeBackend.h src/native/NativeBackend.cpp include/detail/native/NativeRuntime.h src/native/NativeRuntime.cpp include/detail/msp/MSPEngine.h include/detail/KernelArgument.h src/MSPEngine.cpp include/detail/common/LLVMHelper.h src/cuda/CUDADeviceBuffer.cpp include/Promise.h src/native/NativeKernel.cpp include/detail/native/NativeKernel.h include/detail/native/NativeDeviceBuffer.h src/native/NativeDeviceBuffer.cpp )

add_custom_command(TARGET pacxxrt2 POST_BUILD COMMAND $ENV{PACXX_PATH}/bin/llvm-as ${CMAKE_SOURCE_DIR}/src/bitcode/syscalls.ll -o ${CMAKE_BINARY_DIR}/syscalls.bc)
add_custom_command(TARGET pacxxrt2 POST_BUILD COMMAND $ENV{PACXX_PATH}/bin/clang ${PACXX_DEV_FLAGS} ${CMAKE_SOURCE_DIR}/src/bitcode/nvvm_device_binding.cpp -o ${CMAKE_BINARY_DIR}/nvvm_device_binding.bc)
add_custom_command(TARGET pacxxrt2 POST_BUILD COMMAND cp ${CMAKE_SOURCE_DIR}/src/*.S ${CMAKE_BINARY_DIR}/)
#add_custom_command(TARGET pacxxrt2 POST_BUILD COMMAND $ENV{PACXX_PATH}/bin/clang ${PACXX_DEV_FLAGS} ${CMAKE_SOURCE_DIR}/src/DeviceCode.cpp -o ${CMAKE_BINARY_DIR}/DeviceCode.bc)

file(GLOB PACXX_FILES ${CMAKE_BINARY_DIR}/*.a ${CMAKE_BINARY_DIR}/*.bc ${CMAKE_BINARY_DIR}/*.S)

install (FILES ${PACXX_FILES} DESTINATION build_rt)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION $ENV{PACXX_PATH}/include/pacxx)


