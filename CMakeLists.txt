cmake_minimum_required(VERSION 3.0)
project(pacxxrt2)

find_package(CUDA)

if (NOT WIN32)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/pacxx/FindTBB.cmake)
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

set(SOURCE_FILES include/pacxx/Executor.h
        include/pacxx/detail/common/Exceptions.h
        include/pacxx/detail/common/Log.h
        include/pacxx/detail/common/transforms/PACXXTransforms.h
        include/pacxx/detail/common/transforms/ModuleHelper.h
        include/pacxx/detail/common/transforms/CallVisitor.h
        lib/common/transforms/MemoryCoalescing.cpp
        lib/common/transforms/IntrinsicMapper.cpp
        lib/common/transforms/MSPRemover.cpp
        lib/common/transforms/TargetSelection.cpp
        include/pacxx/detail/common/Common.h
        lib/common/Common.cpp
        include/pacxx/detail/CoreInitializer.h
        lib/CoreInitializer.cpp
        include/pacxx/ModuleLoader.h
        lib/ModuleLoader.cpp
        include/pacxx/detail/IRRuntime.h
        include/pacxx/detail/KernelConfiguration.h
        lib/KernelConfiguration.cpp
        include/pacxx/detail/Kernel.h
        include/pacxx/detail/DeviceBuffer.h
        include/pacxx/detail/common/Meta.h
        lib/common/Log.cpp
        lib/Executor.cpp
        include/pacxx/detail/common/Timing.h
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
        include/pacxx/detail/native/NativeBackend.h
        lib/native/NativeBackend.cpp
        include/pacxx/detail/native/NativeRuntime.h
        lib/native/NativeRuntime.cpp
        include/pacxx/detail/msp/MSPEngine.h
        include/pacxx/detail/KernelArgument.h
        lib/MSPEngine.cpp
        include/pacxx/detail/common/LLVMHelper.h
        include/pacxx/Promise.h
        lib/native/NativeKernel.cpp
        include/pacxx/detail/native/NativeKernel.h
        include/pacxx/detail/native/NativeDeviceBuffer.h
        lib/native/NativeDeviceBuffer.cpp
        include/pacxx/pacxx_config.h
        lib/Kernel.cpp
        lib/cuda/PTXBackend.cpp
        include/pacxx/detail/cuda/PTXBackend.h
            lib/common/transforms/AddressSpaceTransform.cpp
            lib/cuda/transforms/NVPTXPrepair.cpp
            lib/rocm/transforms/AMDGCNPrepair.cpp
        lib/IRRuntime.cpp
        lib/native/NativeEvent.cpp
        lib/common/TearDown.cpp)


if (PACXX_USE_CUDA)
    find_package(CUDA REQUIRED)
    set(PACXX_CUDA_BACKEND 1)
    include_directories(${CUDA_TOOLKIT_INCLUDE})
    set(CUDA_SOURCE_FILES
            include/pacxx/detail/cuda/CUDARuntime.h
            lib/cuda/CUDARuntime.cpp
            include/pacxx/detail/cuda/CUDAErrorDetection.h
            lib/cuda/CUDAErrorDetection.cpp
            include/pacxx/detail/cuda/CUDAKernel.h
            lib/cuda/CUDAKernel.cpp
            include/pacxx/detail/cuda/CUDADeviceBuffer.h
            include/pacxx/detail/cuda/CUDAUtility.h
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
        include/pacxx/detail/rocm/HIPRuntime.h
        lib/rocm/HIPRuntime.cpp
        include/pacxx/detail/rocm/HSACOBackend.h
        lib/rocm/HSACOBackend.cpp
        include/pacxx/detail/rocm/HIPErrorDetection.h
        lib/rocm/HIPErrorDetection.cpp
        include/pacxx/detail/rocm/HIPKernel.h
        lib/rocm/HIPKernel.cpp
        include/pacxx/detail/rocm/HIPDeviceBuffer.h
        lib/rocm/HIPDeviceBuffer.cpp
        lib/rocm/HIPEvent.cpp)
set(HIP_REQUIRED ON)

set(HIP_LINK_LIBRARIES "-lhsa-runtime64 -lhip_hcc -lhc_am")

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

set(PACXX_DEV_FLAGS -std=c++14 -emit-llvm -c -O3 -D__device_code__ -isystem ${CMAKE_CURRENT_SOURCE_DIR}/include)



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
            DEPENDS ${nvdbFile}.bc
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
    DEPENDS ${amddbFile}.bc
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

add_library(${PROJECT_NAME} ${SOURCE_FILES} ${CUDA_SOURCE_FILES} ${HIP_SOURCE_FILES})

add_library(pacxx_main STATIC lib/common/WrappedMain.cpp)
add_dependencies(pacxx_main ${PROJECT_NAME})

add_dependencies(${PROJECT_NAME} RV)

add_dependencies(${nvdbFile}_target clang)
add_dependencies(${nadbFile}_target clang)

add_dependencies(${PROJECT_NAME} ${nvdbFile}_target)
add_dependencies(${PROJECT_NAME} ${nadbFile}_target)


add_custom_command(TARGET ${PROJECT_NAME}
        POST_BUILD
        COMMAND ${CMAKE_BINARY_DIR}/bin/llvm-as ${CMAKE_CURRENT_SOURCE_DIR}/lib/bitcode/syscalls.ll -o ${CMAKE_BINARY_DIR}/syscalls.bc)

if (PACXX_USE_HIP)
  target_compile_definitions(${PROJECT_NAME} PUBLIC -D__HIP_PLATFORM_HCC__)
endif()

if(PACXX_USE_OpenMP)
    target_compile_definitions(${PROJECT_NAME} PUBLIC -D__PACXX_OMP PUBLIC -DPACXX_DISABLE_TBB)
endif()

#target_compile_definitions(${PROJECT_NAME} PUBLIC -D__PACXX_RUNTIME_LINKING)

#add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/lib/asm/embed_/*.S ${CMAKE_BINARY_DIR}/)


file(GLOB PACXX_FILES ${CMAKE_BINARY_DIR}/*.S)

install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(TARGETS pacxx_main DESTINATION lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib/asm/ DESTINATION lib)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION include)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cmake/pacxx/ DESTINATION lib/cmake/pacxx)

target_link_libraries(${PROJECT_NAME} PUBLIC ${CUDA_LINK_LIBRARIES} PUBLIC ${PACXX_LLVM_LIBS} PUBLIC ${PACXX_LLVM_SYS_LIBS} PUBLIC ${TBB_LIBRARIES} PUBLIC ${nvdbFile} PUBLIC ${amddbFile} PUBLIC ${nadbFile} PUBLIC ${PACXX_RV_LIB} PUBLIC ${LLD_LIBS})
