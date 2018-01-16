cmake_minimum_required(VERSION 3.0)
project(pacxxrt2)

find_package(CUDA)

if (NOT WIN32)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/pacxx/FindTBB.cmake)
endif()

if (EXISTS /opt/rocm)
set (ROCM_DIR /opt/rocm)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

include_directories(${LLVM_INCLUDE_DIRS})
if (TBB_INCLUDE_DIRS)
include_directories(${TBB_INCLUDE_DIRS})
endif()

if (NOT DEFINED RV_INCLUDE_DIR)
    find_file(RV_HEADER rv.h PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/rv/include/rv
              DOC "RV header file" NO_DEFAULT_PATH)

    if (EXISTS ${RV_HEADER})
        mark_as_advanced(RV_HEADER)
        set(RV_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/rv/include)
    else ()
        message(FATAL_ERROR "RV Header could not be found!")
    endif ()
endif()

include_directories(${RV_INCLUDE_DIR})

set(ENABLE_OPENMP ON)
if (TBB_FOUND)
    set(ENABLE_OPENMP OFF)
endif ()


option(PACXX_USE_OpenMP "use OpenMP instead of TBB" ${ENABLE_OPENMP})
option(PACXX_USE_CUDA "Build the CUDA backend" ${CUDA_FOUND})
option(PACXX_USE_HIP "Build the ROCM/HIP backend" ${HIP_FOUND})

if(PACXX_USE_OpenMP)
    add_compile_options(-fopenmp)
    set(OpenMP_REQUIRED ON)
    set(TBB_REQUIRED OFF)
else ()
    set(OpenMP_REQUIRED OFF)
    set(TBB_REQUIRED ON)
endif(PACXX_USE_OpenMP)

#option(BUILD_SHARED "Build PACXX Runtime as shared object" OFF)
set(PACXX_RV_LIB RV)

set(TRANSFORMS_SOURCE_FILES         
lib/common/transforms/MemoryCoalescing.cpp
lib/common/transforms/IntrinsicMapper.cpp
lib/common/transforms/MSPRemover.cpp
lib/common/transforms/TargetSelection.cpp
lib/common/transforms/MSPGeneration.cpp
lib/common/transforms/MSPCleanup.cpp
lib/common/transforms/IntrinsicScheduler.cpp
lib/common/transforms/CodeGenPrepare.cpp
lib/common/transforms/LoadMotion.cpp
lib/native/transforms/LivenessAnalysis.h
lib/native/transforms/LivenessAnalysis.cpp
lib/native/transforms/BarrierGeneration.cpp
lib/native/transforms/KernelLinker.cpp
lib/native/transforms/MaskedMemTransform.cpp
lib/native/transforms/SMGeneration.h
lib/native/transforms/SMGeneration.cpp
lib/native/transforms/SPMDVectorizer.cpp
lib/common/transforms/ModuleHelper.cpp
lib/common/transforms/AddressSpaceTransform.cpp
lib/cuda/transforms/NVPTXPrepair.cpp
lib/rocm/transforms/AMDGCNPrepair.cpp)

set(SOURCE_FILES 
        lib/common/Common.cpp
        lib/CoreInitializer.cpp
        lib/ModuleLoader.cpp
        lib/KernelConfiguration.cpp
        lib/common/Log.cpp
        lib/Executor.cpp
        lib/native/NativeBackend.cpp
        lib/native/NativeRuntime.cpp
        lib/MSPEngine.cpp
        lib/native/NativeKernel.cpp
        lib/native/NativeDeviceBuffer.cpp
        lib/Kernel.cpp
        lib/cuda/PTXBackend.cpp
        lib/IRRuntime.cpp
        lib/native/NativeEvent.cpp
        lib/common/TearDown.cpp)


if (PACXX_USE_CUDA)
    find_package(CUDA REQUIRED)
    set(PACXX_CUDA_BACKEND 1)
    include_directories(${CUDA_TOOLKIT_INCLUDE})
    set(CUDA_SOURCE_FILES
            lib/cuda/CUDARuntime.cpp
            lib/cuda/CUDAErrorDetection.cpp
            lib/cuda/CUDAKernel.cpp
            lib/cuda/CUDADeviceBuffer.cpp
            lib/cuda/CUDAEvent.cpp)
    set(CUDA_REQUIRED ON)

    set (CUDA_LINK_LIBRARIES ${CUDA_LIBRARIES})
    #   set(CUDA_LINK_LIBRARIES "-lcuda -lcudart")

    if (BUILD_SHARED)
        set(CUDA_USE_SHARED_RT ON)
        set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE BOOL "" FORCE) # unfortunately, this gets overwritten in FindCUDA.cmake, so we have to force set it again
    else ()
        set(CUDA_USE_SHARED_RT OFF)
    endif ()

else (PACXX_USE_CUDA)
    set(CUDA_REQUIRED OFF)
    set(PACXX_CUDA_BACKEND 0)
endif (PACXX_USE_CUDA)


if (PACXX_USE_HIP)
	if (NOT ROCM_DIR)
		message(FATAL_ERROR
				"ROCm - Not found! (please set ROCM_DIR)")
	else ()
		message(STATUS "ROCm - Found")
	endif ()
set(PACXX_HIP_BACKEND 1)
include_directories(${ROCM_DIR}/include)
link_directories(${ROCM_DIR}/lib)
set(HIP_SOURCE_FILES
        lib/rocm/HIPRuntime.cpp
        lib/rocm/HSACOBackend.cpp
        lib/rocm/HIPErrorDetection.cpp
        lib/rocm/HIPKernel.cpp
        lib/rocm/HIPDeviceBuffer.cpp
        lib/rocm/HIPEvent.cpp)
set(HIP_REQUIRED ON)

find_file(ASIO_HEADER asio.hpp PATHS /usr/include)

if (EXISTS ${ASIO_HEADER})
    message(STATUS "ASIO found")
    set(REMOTE_SOURCE_FILES
        lib/remote/RemoteRuntime.cpp
        lib/remote/RemoteKernel.cpp
        lib/remote/RemoteDeviceBuffer.cpp
        lib/remote/RemoteEvent.cpp)
else()
    message(FATAL_ERROR "ASIO not found")
endif()

set(HIP_LINK_LIBRARIES hsa-runtime64 hip_hcc hc_am)

set(LLD_LIBS lldCOFF lldDriver lldELF lldMinGW)

else (PACXX_USE_HIP)
set(HIP_REQUIRED OFF)
set(PACXX_HIP_BACKEND 0)
endif (PACXX_USE_HIP)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config/config.h.in ${CMAKE_CURRENT_SOURCE_DIR}/include/pacxx/pacxx_config.h)
configure_file(cmake/config/FindPACXXConfig.cmake.in ${CMAKE_CURRENT_SOURCE_DIR}/cmake/pacxx/FindPACXXConfig.cmake)


set(CMAKE_CXX_STANDARD 14)

add_compile_options(-include ${CMAKE_CURRENT_SOURCE_DIR}/include/pacxx/pacxx_config.h -Wno-attributes)

#add_definitions(${LLVM_DEFINITIONS} -D__PACXX_RUNTIME_LINKING)
add_definitions(${LLVM_DEFINITIONS})

set(PACXX_DEV_FLAGS -std=c++14 -pacxx -emit-llvm -c -O3 -D__device_code__ -isystem ${CMAKE_CURRENT_SOURCE_DIR}/include)



if (${CMAKE_SIZEOF_VOID_P} EQUAL 8)
    set(PACXX_CUDA_DEV_FLAGS ${PACXX_DEV_FLAGS} -target nvptx64-unknown-unknown)
else ()
    set(PACXX_CUDA_DEV_FLAGS ${PACXX_DEV_FLAGS} -target nvptx-unknown-unknown)
endif ()

# generate the nvptx device binding
set(nvdbFile nvptx_device_binding)

add_custom_command(
        OUTPUT ${nvdbFile}.bc
        COMMAND ${CMAKE_BINARY_DIR}/bin/clang ${PACXX_DEV_FLAGS}  -isystem ${CUDA_TOOLKIT_INCLUDE} -I ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/lib/bitcode/${nvdbFile}.cpp -o ${nvdbFile}.bc
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        DEPENDS clang
        COMMENT "Generating NVPTX Device Binding")

        add_custom_command(
            OUTPUT ${nvdbFile}.linked.bc
            COMMAND ${CMAKE_BINARY_DIR}/bin/llvm-link -suppress-warnings ${nvdbFile}.bc ${CMAKE_CURRENT_SOURCE_DIR}/lib/bitcode/syscalls.ll ${CMAKE_CURRENT_SOURCE_DIR}/lib/bitcode/nvptx_libdevice.bc -o ${nvdbFile}.linked.bc
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/lib    
            DEPENDS ${nvdbFile}.bc llvm-link
            COMMENT "Linking NVPTX libdevice into bitcode")

add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/lib/${nvdbFile}.o
        COMMAND ${CMAKE_BINARY_DIR}/bin/clang -DFILE='"${nvdbFile}.linked.bc"' -DTAG='nvptx_binding' ${CMAKE_CURRENT_SOURCE_DIR}/lib/asm/embed.S -c -o ${nvdbFile}.o
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        DEPENDS ${nvdbFile}.linked.bc
        COMMENT "Preparing NVPTX Device Binding for linking")

add_custom_target(${nvdbFile}_target DEPENDS ${nvdbFile}.bc ${CMAKE_BINARY_DIR}/lib/${nvdbFile}.o)

add_library(${nvdbFile} STATIC ${CMAKE_BINARY_DIR}/lib/${nvdbFile}.o)


set_source_files_properties(${CMAKE_BINARY_DIR}/lib/${nvdbFile}.o PROPERTIES EXTERNAL_OBJECT true GENERATED true)
set_target_properties(${nvdbFile} PROPERTIES LINKER_LANGUAGE C)

# generate the amdgcn device binding
set(amddbFile amdgcn_device_binding)

add_custom_command(
        OUTPUT ${amddbFile}.bc
        COMMAND ${CMAKE_BINARY_DIR}/bin/clang ${PACXX_DEV_FLAGS} -isystem ${CUDA_TOOLKIT_INCLUDE} -I ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/lib/bitcode/${amddbFile}.cpp -o ${amddbFile}.bc
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        DEPENDS clang
        COMMENT "Generating AMDGCN Device Binding")

add_custom_command(
    OUTPUT ${amddbFile}.linked.bc
    COMMAND ${CMAKE_BINARY_DIR}/bin/llvm-link -suppress-warnings ${amddbFile}.bc ${CMAKE_CURRENT_SOURCE_DIR}/lib/bitcode/amdgcn_libdevice.bc -o ${amddbFile}.linked.bc
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/lib    
    DEPENDS ${amddbFile}.bc llvm-link
    COMMENT "Linking AMDGCN libdevice into bitcode")

add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/lib/${amddbFile}.o
        COMMAND ${CMAKE_BINARY_DIR}/bin/clang -DFILE='"${amddbFile}.linked.bc"' -DTAG='amdgcn_binding' ${CMAKE_CURRENT_SOURCE_DIR}/lib/asm/embed.S -c -o ${amddbFile}.o
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        DEPENDS ${amddbFile}.linked.bc
        COMMENT "Preparing AMDGCN Device Binding for linking")

add_custom_target(${amddbFile}_target DEPENDS ${amddbFile}.bc ${CMAKE_BINARY_DIR}/lib/${amddbFile}.o)

add_library(${amddbFile} STATIC ${CMAKE_BINARY_DIR}/lib/${amddbFile}.o)


set_source_files_properties(${CMAKE_BINARY_DIR}/lib/${amddbFile}.o PROPERTIES EXTERNAL_OBJECT true GENERATED true)
set_target_properties(${amddbFile} PROPERTIES LINKER_LANGUAGE C)


# generate the native device binding
set(nadbFile native_device_binding)

add_custom_command(
        OUTPUT ${nadbFile}.bc
        COMMAND ${CMAKE_BINARY_DIR}/bin/clang ${PACXX_DEV_FLAGS} -I/usr/include ${CMAKE_CURRENT_SOURCE_DIR}/lib/bitcode/${nadbFile}.cpp -o ${nadbFile}.bc
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        DEPENDS clang
        COMMENT "Generating Native Device Binding")

add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/lib/${nadbFile}.o
        COMMAND ${CMAKE_BINARY_DIR}/bin/clang -DFILE='"${nadbFile}.bc"' -DTAG='native_binding' ${CMAKE_CURRENT_SOURCE_DIR}/lib/asm/embed.S -c -o ${nadbFile}.o
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        DEPENDS ${nadbFile}.bc
        COMMENT "Preparing Native Device Binding for linking")

add_custom_target(${nadbFile}_target DEPENDS ${nadbFile}.bc ${CMAKE_BINARY_DIR}/lib/${nadbFile}.o)
add_library(${nadbFile} STATIC ${CMAKE_BINARY_DIR}/lib/${nadbFile}.o)

set_source_files_properties(${CMAKE_BINARY_DIR}/lib/${nadbFile}.o PROPERTIES EXTERNAL_OBJECT true GENERATED true)
set_target_properties(${nadbFile} PROPERTIES LINKER_LANGUAGE C)

link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64) #TODO make dynamic for non 64 bit systems
execute_process(COMMAND ${CMAKE_BINARY_DIR}/bin/llvm-config --libfiles
        OUTPUT_VARIABLE PACXX_LLVM_LIBS
        RESULT_VARIABLE PACXX_LLVM_CONFIG_RESULT OUTPUT_STRIP_TRAILING_WHITESPACE)
separate_arguments(PACXX_LLVM_LIBS)

execute_process(COMMAND ${CMAKE_BINARY_DIR}/bin/llvm-config --system-libs
        OUTPUT_VARIABLE PACXX_LLVM_SYS_LIBS
        RESULT_VARIABLE PACXX_LLVM_CONFIG_RESULT OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND ${CMAKE_BINARY_DIR}/bin/llvm-config --ldflags
        OUTPUT_VARIABLE PACXX_LD_FLAGS
        RESULT_VARIABLE PACXX_LLVM_CONFIG_RESULT OUTPUT_STRIP_TRAILING_WHITESPACE)

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--allow-shlib-undefined")

# main runtime lib
add_library(${PROJECT_NAME} ${SOURCE_FILES} ${REMOTE_SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC PACXXTransforms PUBLIC ${PACXX_LLVM_LIBS} PUBLIC ${PACXX_LLVM_SYS_LIBS} PUBLIC ${TBB_LIBRARIES} PUBLIC ${nadbFile})

#cuda device back end
if (PACXX_USE_CUDA)
add_library(PACXXBeCUDA ${CUDA_SOURCE_FILES})
target_link_libraries(PACXXBeCUDA PUBLIC ${CUDA_LINK_LIBRARIES} PUBLIC ${PACXX_LLVM_LIBS} PUBLIC ${PACXX_LLVM_SYS_LIBS} PUBLIC ${nvdbFile})
install(TARGETS PACXXBeCUDA DESTINATION lib)
target_link_libraries(${PROJECT_NAME} PUBLIC PACXXBeCUDA)
endif()

#rocm device back end
if (PACXX_USE_HIP)
add_library(PACXXBeROCm ${HIP_SOURCE_FILES})
target_link_libraries(PACXXBeROCm PUBLIC ${HIP_LINK_LIBRARIES} PUBLIC ${LLD_LIBS} PUBLIC ${PACXX_LLVM_LIBS} PUBLIC ${PACXX_LLVM_SYS_LIBS} PUBLIC ${amddbFile})
target_compile_definitions(PACXXBeROCm PUBLIC -D__HIP_PLATFORM_HCC__)
install(TARGETS PACXXBeROCm DESTINATION lib)
target_link_libraries(${PROJECT_NAME} PUBLIC PACXXBeROCm)
endif()

#llvm transformations
add_library(PACXXTransforms ${TRANSFORMS_SOURCE_FILES})
target_link_libraries(PACXXTransforms PUBLIC ${PACXX_LLVM_LIBS} PUBLIC ${PACXX_LLVM_SYS_LIBS} PUBLIC ${PACXX_RV_LIB})
install(TARGETS PACXXTransforms DESTINATION lib)


add_library(pacxx_main STATIC lib/common/WrappedMain.cpp)
add_dependencies(pacxx_main ${PROJECT_NAME})

add_dependencies(${PROJECT_NAME} RV)

add_custom_command(TARGET ${PROJECT_NAME}
        POST_BUILD
        COMMAND ${CMAKE_BINARY_DIR}/bin/llvm-as ${CMAKE_CURRENT_SOURCE_DIR}/lib/bitcode/syscalls.ll -o ${CMAKE_BINARY_DIR}/syscalls.bc)



if(PACXX_USE_OpenMP)
    target_compile_definitions(${PROJECT_NAME} PUBLIC -D__PACXX_OMP PUBLIC -DPACXX_DISABLE_TBB)
endif()

file(GLOB PACXX_FILES ${CMAKE_BINARY_DIR}/*.S)

install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(TARGETS pacxx_main DESTINATION lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib/asm/ DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cmake/pacxx/ DESTINATION lib/cmake/pacxx)

