cmake_minimum_required(VERSION 3.0)
project(pacxxrt2)

set(CMAKE_MODULE_PATH $ENV{PACXX_PATH}/lib/cmake/llvm/)
set(LLVM_DIR $ENV{PACXX_PATH}/lib/cmake/llvm/)
find_package(LLVM REQUIRED CONFIG)
find_package(CUDA)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/pacxx/FindTBB.cmake)

include_directories("./include")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${TBB_INCLUDE_DIR})

set(ENABLE_OPENMP ON)
if (TBB_FOUND)
    set(ENABLE_OPENMP OFF)
endif ()


option(OpenMP "use OpenMP instead of TBB" ${ENABLE_OPENMP})
option(USE_CUDA "Build the CUDA backend" ${CUDA_FOUND})

if(OpenMP)
  add_definitions(-D__PACXX_OMP)
  add_definitions(-DPACXX_DISABLE_TBB)
  add_compile_options(-fopenmp)
  set(OpenMP_REQUIRED ON)
  set(TBB_REQUIRED OFF)
else ()
    set(OpenMP_REQUIRED OFF)
    set(TBB_REQUIRED ON)
endif(OpenMP)


option(BUILD_SHARED "Build PACXX Runtime as shared object" OFF)


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
        src/Kernel.cpp
        src/IRRuntime.cpp)


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
    set(CUDA_REQUIRED ON)

    if (BUILD_SHARED)
        set(CUDA_USE_SHARED_RT ON)
        set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE BOOL "" FORCE) # unfortunately, this gets overwritten in FindCUDA.cmake, so we have to force set it again
    else ()
        set(CUDA_USE_SHARED_RT OFF)
    endif ()

else (USE_CUDA)
    set(CUDA_REQUIRED OFF)
    set(PACXX_CUDA_BACKEND 0)
endif (USE_CUDA)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config/config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/include/pacxx/config.h)
configure_file(cmake/config/FindPACXXConfig.cmake.in ${CMAKE_CURRENT_SOURCE_DIR}/cmake/pacxx/FindPACXXConfig.cmake)


set(CMAKE_CXX_STANDARD 14)

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
        OUTPUT ${nvdbFile}.o
        COMMAND ${LLVM_TOOLS_BINARY_DIR}/clang -DFILE='"${CMAKE_BINARY_DIR}/${nvdbFile}.bc"' ${CMAKE_SOURCE_DIR}/src/asm/embed_runtime/embed_nvptx_binding.S -c -o ${nvdbFile}.o
        WORKING_DIRECTORY ${binDir}
        DEPENDS ${nvdbFile}.bc
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
        COMMAND ${LLVM_TOOLS_BINARY_DIR}/clang -DFILE='"${CMAKE_BINARY_DIR}/${nadbFile}.bc"' ${CMAKE_SOURCE_DIR}/src/asm/embed_runtime/embed_native_binding.S -c -o ${nadbFile}.o
        WORKING_DIRECTORY ${binDir}
        DEPENDS ${nadbFile}.bc
        COMMENT "Preparing Native Device Binding for linking")

add_custom_target(${nadbFile}_target DEPENDS ${nadbFile}.bc ${nadbFile}.o)
add_library(${nadbFile} STATIC ${nadbFile}.o)

set_source_files_properties(${nadbFile}.o PROPERTIES EXTERNAL_OBJECT true GENERATED true)
set_target_properties(${nadbFile} PROPERTIES LINKER_LANGUAGE C)

if (BUILD_SHARED)
    add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${CUDA_SOURCE_FILES} ${nvdbFile}.o ${nadbFile}.o)
else ()
    add_library(${PROJECT_NAME} ${SOURCE_FILES} ${CUDA_SOURCE_FILES} ${nvdbFile}.o ${nadbFile}.o)
endif ()

add_dependencies(${PROJECT_NAME} ${nvdbFile}_target)
add_dependencies(${PROJECT_NAME} ${nadbFile}_target)


add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND ${LLVM_TOOLS_BINARY_DIR}/llvm-as ${CMAKE_SOURCE_DIR}/src/bitcode/syscalls.ll -o ${CMAKE_BINARY_DIR}/syscalls.bc)

#add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND cp ${CMAKE_SOURCE_DIR}/src/asm/embed_/*.S ${CMAKE_BINARY_DIR}/)


file(GLOB PACXX_FILES ${CMAKE_BINARY_DIR}/*.S)


if (BUILD_SHARED)
    install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)
else ()
    install(TARGETS ${PROJECT_NAME} ARCHIVE DESTINATION lib)
endif ()


install(DIRECTORY ${CMAKE_SOURCE_DIR}/src/asm/embed_executable/ DESTINATION lib)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/cmake/pacxx/ DESTINATION lib/cmake/pacxx)


