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
        include/pacxx/config.h
        src/Kernel.cpp src/IRRuntime.cpp)


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
    set(PACXX_CUDA_DEV_FLAGS ${PACXX_DEV_FLAGS} -target nvptx64-unknown-unknown)
else ()
    set(PACXX_CUDA_DEV_FLAGS ${PACXX_DEV_FLAGS} -target nvptx-unknown-unknown)
endif ()

# generate the nvptx device binding
set(nvdbFile nvptx_device_binding)

add_custom_command(
        OUTPUT ${nvdbFile}.bc
        COMMAND ${LLVM_TOOLS_BINARY_DIR}/clang ${PACXX_CUDA_DEV_FLAGS} ${CMAKE_SOURCE_DIR}/src/bitcode/${nvdbFile}.cpp -o ${CMAKE_BINARY_DIR}/${nvdbFile}.bc
        WORKING_DIRECTORY ${binDir}
        COMMENT "Generating NVPTX Device Binding")

add_custom_command(
        OUTPUT ${ndbFile}.o
        COMMAND ${LLVM_TOOLS_BINARY_DIR}/clang -DFILE='"${CMAKE_BINARY_DIR}/${nvdbFile}.bc"' ${CMAKE_SOURCE_DIR}/src/embed_nvptx_binding.S -c -o ${nvdbFile}.o
        WORKING_DIRECTORY ${binDir}
        DEPENDS ${ndbFile}.bc
        COMMENT "Preparing NVPTX Device Binding for linking")

add_custom_target(${nvdbFile}_target DEPENDS ${nvdbFile}.bc ${nvdbFile}.o)

add_library(${nvdbFile} STATIC ${nvdbFile}.o)


set_source_files_properties(${nvdbFile}.o PROPERTIES EXTERNAL_OBJECT true GENERATED true)
set_target_properties(${nvdbFile} PROPERTIES LINKER_LANGUAGE C)


# generate the native device binding
set(nadbFile native_device_binding)

add_custom_command(
        OUTPUT ${nadbFile}.bc
        COMMAND ${LLVM_TOOLS_BINARY_DIR}/clang ${PACXX_DEV_FLAGS} ${CMAKE_SOURCE_DIR}/src/bitcode/${nadbFile}.cpp -o ${CMAKE_BINARY_DIR}/${nadbFile}.bc
        WORKING_DIRECTORY ${binDir}
        COMMENT "Generating Native Device Binding")

add_custom_command(
        OUTPUT ${nadbFile}.o
        COMMAND ${LLVM_TOOLS_BINARY_DIR}/clang -DFILE='"${CMAKE_BINARY_DIR}/${nadbFile}.bc"' ${CMAKE_SOURCE_DIR}/src/embed_native_binding.S -c -o ${nadbFile}.o
        WORKING_DIRECTORY ${binDir}
        DEPENDS ${nadbFile}.bc
        COMMENT "Preparing Native Device Binding for linking")

add_custom_target(${nadbFile}_target DEPENDS ${nadbFile}.bc ${nadbFile}.o)
add_library(${nadbFile} STATIC ${nadbFile}.o)

set_source_files_properties(${nadbFile}.o PROPERTIES EXTERNAL_OBJECT true GENERATED true)
set_target_properties(${nadbFile} PROPERTIES LINKER_LANGUAGE C)


add_library(pacxxrt2 ${SOURCE_FILES} ${CUDA_SOURCE_FILES} ${nvdbFile}.o ${nadbFile}.o)

add_dependencies(pacxxrt2 ${nvdbFile}_target)
add_dependencies(pacxxrt2 ${nadbFile}_target)


add_custom_command(TARGET pacxxrt2 POST_BUILD COMMAND ${LLVM_TOOLS_BINARY_DIR}/llvm-as ${CMAKE_SOURCE_DIR}/src/bitcode/syscalls.ll -o ${CMAKE_BINARY_DIR}/syscalls.bc)

add_custom_command(TARGET pacxxrt2 POST_BUILD COMMAND cp ${CMAKE_SOURCE_DIR}/src/*.S ${CMAKE_BINARY_DIR}/)


file(GLOB PACXX_FILES ${CMAKE_BINARY_DIR}/*.a ${CMAKE_BINARY_DIR}/*.bc ${CMAKE_BINARY_DIR}/*.S)

install(FILES ${PACXX_FILES} DESTINATION lib)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/cmake/ DESTINATION lib/cmake)


