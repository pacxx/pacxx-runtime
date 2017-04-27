cmake_minimum_required(VERSION 3.0)
project(PACXX_V2)

set(CMAKE_MODULE_PATH $ENV{PACXX_PATH}/lib/cmake/llvm/)
set(LLVM_DIR $ENV{PACXX_PATH}/lib/cmake/llvm/)
find_package(LLVM REQUIRED CONFIG)
find_package(CUDA)

include_directories("./include")

set(TBB_INCLUDE_DIR "$ENV{PACXX_PATH}/include" CACHE FILEPATH "include directory of tbb")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${TBB_INCLUDE_DIR})

option(OpenMP "use OpenMP instead of TBB" OFF)
option(USE_CUDA "Build the CUDA backend" ${CUDA_FOUND})

if(OpenMP)
  add_definitions(-D__PACXX_OMP)
  add_definitions(-DPACXX_DISABLE_TBB)
  add_compile_options(-fopenmp)
endif(OpenMP)


set(SOURCE_FILES include/pacxx/Executor.h
        include/pacxx/detail/common/Exceptions.h
        include/pacxx/detail/common/Log.h
        include/pacxx/detail/common/Common.h
        src/Common.cpp
        include/pacxx/detail/CoreInitializer.h
        src/CoreInitializer.cpp
        include/pacxx/ModuleLoader.h
        src/ModuleLoader.cpp
        include/pacxx/detail/IRRuntime.h
        include/pacxx/CodePolicy.h
        include/pacxx/detail/KernelConfiguration.h
        src/KernelConfiguration.cpp
        include/pacxx/detail/Kernel.h
        include/pacxx/detail/DeviceBuffer.h
        include/pacxx/detail/MemoryManager.h
        include/pacxx/detail/common/Meta.h
        src/Log.cpp
        src/Executor.cpp
        include/pacxx/detail/common/Timing.h
        include/pacxx/detail/native/NativeBackend.h
        src/native/NativeBackend.cpp
        include/pacxx/detail/native/NativeRuntime.h
        src/native/NativeRuntime.cpp
        include/pacxx/detail/msp/MSPEngine.h
        include/pacxx/detail/KernelArgument.h
        src/MSPEngine.cpp
        include/pacxx/detail/common/LLVMHelper.h
        include/pacxx/Promise.h
        src/native/NativeKernel.cpp
        include/pacxx/detail/native/NativeKernel.h
        include/pacxx/detail/native/NativeDeviceBuffer.h
        src/native/NativeDeviceBuffer.cpp
        include/pacxx/config.h)


if (USE_CUDA)
    find_package(CUDA REQUIRED)
    set(PACXX_CUDA_BACKEND 1)
    include_directories(${CUDA_TOOLKIT_INCLUDE})
    set(CUDA_SOURCE_FILES src/cuda/PTXBackend.cpp
            include/pacxx/detail/cuda/PTXBackend.h
            include/pacxx/detail/cuda/CUDARuntime.h
            src/cuda/CUDARuntime.cpp
            include/pacxx/detail/cuda/CUDAErrorDetection.h
            src/cuda/CUDAErrorDetection.cpp
            include/pacxx/detail/cuda/CUDAKernel.h
            src/cuda/CUDAKernel.cpp
            include/pacxx/detail/cuda/CUDADeviceBuffer.h
            include/pacxx/detail/cuda/CUDAUtility.h
            src/cuda/CUDADeviceBuffer.cpp)
else (USE_CUDA)
    set(PACXX_CUDA_BACKEND 0)
endif (USE_CUDA)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config/config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/include/pacxx/config.h)


add_compile_options(-std=c++14)
add_compile_options(-include ${CMAKE_CURRENT_SOURCE_DIR}/include/pacxx/config.h)

add_definitions(${LLVM_DEFINITIONS})

set(PACXX_DEV_FLAGS -std=pacxx -O0 -c -emit-llvm -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -D__CUDA_DEVICE_CODE -isystem $ENV{PACXX_PATH}/include/c++/v1 -isystem $ENV{PACXX_PATH}include -isystem ${CUDA_TOOLKIT_INCLUDE} -I ${CMAKE_SOURCE_DIR}/include)



if (${CMAKE_SIZEOF_VOID_P} EQUAL 8)
    set(PACXX_DEV_FLAGS ${PACXX_DEV_FLAGS} -target nvptx64-unknown-unknown)
else ()
    set(PACXX_DEV_FLAGS ${PACXX_DEV_FLAGS} -target nvptx-unknown-unknown)
endif ()


set(ndbFile nvptx_device_binding)

add_custom_command(
        OUTPUT ${ndbFile}.bc
        COMMAND ${LLVM_TOOLS_BINARY_DIR}/clang ${PACXX_DEV_FLAGS} ${CMAKE_SOURCE_DIR}/src/bitcode/nvvm_device_binding.cpp -o ${CMAKE_BINARY_DIR}/${ndbFile}.bc
        WORKING_DIRECTORY ${binDir}
        COMMENT "Generating NVPTX Device Binding")

add_custom_command(
        OUTPUT ${ndbFile}.o
        COMMAND ${LLVM_TOOLS_BINARY_DIR}/clang -DFILE='"${CMAKE_BINARY_DIR}/${ndbFile}.bc"' ${CMAKE_SOURCE_DIR}/src/embed_nvptx_binding.S -c -o ${ndbFile}.o
        WORKING_DIRECTORY ${binDir}
        DEPENDS ${ndbFile}.bc
        COMMENT "Preparing NVPTX Device Binding for linking")

add_custom_target(${ndbFile}_target DEPENDS ${ndbFile}.bc ${ndbFile}.o)

add_library(${ndbFile} STATIC ${ndbFile}.o)
add_library(pacxxrt2 ${SOURCE_FILES} ${CUDA_SOURCE_FILES} ${ndbFile}.o)


set_source_files_properties(${ndbFile}.o PROPERTIES EXTERNAL_OBJECT true GENERATED true)
set_target_properties(${ndbFile} PROPERTIES LINKER_LANGUAGE C)

add_dependencies(pacxxrt2 ${ndbFile}_target)

#target_link_libraries(pacxxrt2 ${ndbFile})

add_custom_command(TARGET pacxxrt2 POST_BUILD COMMAND ${LLVM_TOOLS_BINARY_DIR}/llvm-as ${CMAKE_SOURCE_DIR}/src/bitcode/syscalls.ll -o ${CMAKE_BINARY_DIR}/syscalls.bc)
add_custom_command(TARGET pacxxrt2 POST_BUILD COMMAND ${LLVM_TOOLS_BINARY_DIR}/clang ${PACXX_DEV_FLAGS} ${CMAKE_SOURCE_DIR}/src/bitcode/nvvm_device_binding.cpp -o ${CMAKE_BINARY_DIR}/nvvm_device_binding.bc)
add_custom_command(TARGET pacxxrt2 POST_BUILD COMMAND cp ${CMAKE_SOURCE_DIR}/src/*.S ${CMAKE_BINARY_DIR}/)
#add_custom_command(TARGET pacxxrt2 POST_BUILD COMMAND ${LLVM_TOOLS_BINARY_DIR}/clang ${PACXX_DEV_FLAGS} ${CMAKE_SOURCE_DIR}/src/DeviceCode.cpp -o ${CMAKE_BINARY_DIR}/DeviceCode.bc)


file(GLOB PACXX_FILES ${CMAKE_BINARY_DIR}/*.a ${CMAKE_BINARY_DIR}/*.bc ${CMAKE_BINARY_DIR}/*.S)

install(FILES ${PACXX_FILES} DESTINATION lib)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/cmake/ DESTINATION lib/cmake)


