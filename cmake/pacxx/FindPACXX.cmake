cmake_minimum_required(VERSION 3.5)


if (NOT PACXX_DIR)
    message(FATAL_ERROR
            "PACXX - Not found! (please set PACXX_DIR)")
else ()
    message(STATUS "PACXX - Found")
endif ()

if (EXISTS /opt/rocm)
set (ROCM_DIR /opt/rocm)
endif()

set(PACXX_DIR ${PACXX_DIR} CACHE PATH "Path to PACXX")

include(${PACXX_DIR}/lib/cmake/pacxx/FindPACXXConfig.cmake)

if (NOT CUDA_FOUND)
if (CUDA_REQUIRED)
    find_package(CUDA REQUIRED)

    if (CUDA_FOUND)
        include_directories(${CUDA_TOOLKIT_INCLUDE})
        include_directories(${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/include)
        link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64) #TODO make dynamic for non 64 bit systems
    endif ()

    set(CUDA_LINK_LIBRARIES PACXXBeCUDA cuda ${CUDA_cupti_LIBRARY})

endif ()
endif ()

if (HIP_REQUIRED)
	set(ROCM_DIR ${ROCM_DIR} CACHE PATH "Path to ROCm")
	if (NOT ROCM_DIR)
		message(FATAL_ERROR
				"ROCm - Not found! (please set ROCM_DIR)")
	else ()
		message(STATUS "ROCm - Found")
	endif ()
    if (HIP_FOUND)
        include_directories(${ROCM_DIR}/include)
    endif ()

    link_directories(${ROCM_DIR}/lib)
    set(HIP_LINK_LIBRARIES PACXXBeROCm hsa-runtime64 hip_hcc hc_am)
endif()

if (OpenMP_REQUIRED)
    find_package(OpenMP REQUIRED)
    if (OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif ()
endif ()

if (TBB_REQUIRED)
    include(${PACXX_DIR}/lib/cmake/pacxx/FindTBB.cmake)
endif ()

include(${PACXX_DIR}/lib/cmake/pacxx/FindPAPI.cmake)

# Set the path to llvm-config
find_program(PACXX_LLVM_CONFIG llvm-config PATHS
  ${PACXX_DIR} PATH_SUFFIXES bin NO_DEFAULT_PATH)
if (EXISTS ${PACXX_LLVM_CONFIG})
    mark_as_advanced(PACXX_LLVM_CONFIG)
else ()
    message(FATAL_ERROR "llvm-config (PACXX) - Not found! (${PACXX_LLVM_CONFIG})")
endif ()


execute_process(COMMAND ${PACXX_LLVM_CONFIG} --libs
        OUTPUT_VARIABLE PACXX_LLVM_LIBS
        RESULT_VARIABLE PACXX_LLVM_CONFIG_RESULT OUTPUT_STRIP_TRAILING_WHITESPACE)
if (NOT PACXX_LLVM_CONFIG_RESULT EQUAL "0")
    message(FATAL_ERROR "llvm-config - Error geting LLVM Libraries!")
else ()
    mark_as_advanced(PACXX_LLVM_LIBS)
endif ()

execute_process(COMMAND ${PACXX_LLVM_CONFIG} --system-libs
        OUTPUT_VARIABLE PACXX_LLVM_SYS_LIBS
        RESULT_VARIABLE PACXX_LLVM_CONFIG_RESULT OUTPUT_STRIP_TRAILING_WHITESPACE)
if (NOT PACXX_LLVM_CONFIG_RESULT EQUAL "0")
    message(FATAL_ERROR "llvm-config - Error geting LLVM System Libraries!")
else ()
    mark_as_advanced(PACXX_LLVM_SYS_LIBS)
endif ()


execute_process(COMMAND ${PACXX_LLVM_CONFIG} --ldflags
        OUTPUT_VARIABLE PACXX_LD_FLAGS
        RESULT_VARIABLE PACXX_LLVM_CONFIG_RESULT OUTPUT_STRIP_TRAILING_WHITESPACE)
if (NOT PACXX_LLVM_CONFIG_RESULT EQUAL "0")
    message(FATAL_ERROR "llvm-config - Error geting LLVM linker flags!")
else ()
    mark_as_advanced(PACXX_LD_FLAGS)
endif ()


find_program(PACXX_COMPILER clang++ PATHS
  ${PACXX_DIR} PATH_SUFFIXES bin NO_DEFAULT_PATH)
if (EXISTS ${PACXX_COMPILER})
    message(STATUS "clang location: ${PACXX_COMPILER}")
    mark_as_advanced(PACXX_COMPILER)
else ()
    message(FATAL_ERROR "clang++ (PACXX) - Not found! (${PACXX_COMPILER})")
endif ()

find_program(PACXX_LINK llvm-link PATHS
  ${PACXX_DIR} PATH_SUFFIXES bin NO_DEFAULT_PATH)
if (EXISTS ${PACXX_LINK})
    mark_as_advanced(PACXX_LINK)
else ()
    message(FATAL_ERROR "llvm-link (PACXX) - Not found! (${PACXX_LINK})")
endif ()

find_program(PACXX_OPT opt PATHS ${PACXX_DIR}/bin NO_DEFAULT_PATH)
if (EXISTS ${PACXX_OPT})
    message(STATUS "opt location: ${PACXX_OPT}")
    message(STATUS "pacxx: ${PACXX_DIR}")
    mark_as_advanced(PACXX_OPT)
else ()
    message(FATAL_ERROR "opt (PACXX) - Not found! (${PACXX_OPT})")
endif ()


find_library(PACXX_RUNTIME_LIBRARY pacxxrt2 PATHS ${PACXX_DIR}
        HINTS ${PACXX_DIR}/lib PATH_SUFFIXES lib
        DOC "PACXX Runtime Library" NO_DEFAULT_PATH)

if (EXISTS ${PACXX_RUNTIME_LIBRARY})
    mark_as_advanced(PACXX_RUNTIME_LIBRARY)
else ()
    message(FATAL_ERROR "libpacxxrt2 - Not found!")
endif ()

find_library(PACXX_RV_LIBRARY RV PATHS ${PACXX_DIR}
        HINTS ${PACXX_DIR}/lib PATH_SUFFIXES lib
        DOC "Region Vectorizer Library" NO_DEFAULT_PATH)

if (EXISTS ${PACXX_RV_LIBRARY})
    mark_as_advanced(PACXX_RV_LIBRARY)
else ()
    message(FATAL_ERROR "libRV - Not found!")
endif ()


find_file(PACXX_ASM_WRAPPER embed.S PATHS ${PACXX_DIR}
        HINTS ${PACXX_DIR}/lib PATH_SUFFIXES lib
        DOC "PACXX Wrapper Assembly File" NO_DEFAULT_PATH)

if (EXISTS ${PACXX_ASM_WRAPPER})
    mark_as_advanced(PACXX_ASM_WRAPPER)
else ()
    message(FATAL_ERROR "embed.S - Not found!")
endif ()


set(PACXX_INCLUDE_DIRECTORY ${PACXX_DIR}/include)
if (NOT EXISTS ${PACXX_INCLUDE_DIRECTORY})
    message(FATAL_ERROR "PACXX includes - Not found!")
endif ()

set(PACXX_FOUND 1)

set(PACXX_DEVICE_FLAGS "-Wno-invalid-noreturn -std=c++1z -pacxx -O0 -emit-llvm -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -DNDEBUG -D__CUDA_DEVICE_CODE" CACHE "PACXX Device compilation flags" STRING)
set(PACXX_LINK_FLAGS "-suppress-warnings" CACHE "PACXX bytecode linker flags" STRING)

function(pacxx_generate_ir targetName srcFile binDir)

    get_filename_component(srcName ${srcFile} NAME)

    set(outFile ${binDir}/${srcName}.bc)

    get_property(includes DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY
            INCLUDE_DIRECTORIES)
    set(device_includes "")
    foreach (directory ${includes})
        set(device_includes "-I${directory}" ${device_includes})
    endforeach ()
    if (CMAKE_INCLUDE_PATH)
        foreach (directory ${CMAKE_INCLUDE_PATH})
            set(device_includes "-I${directory}"
                    ${device_includes})
        endforeach ()
    endif ()

    #get_target_property(targetDefs ${targetName} COMPILE_DEFINITIONS)
    get_property(targetDefs TARGET ${targetName} PROPERTY COMPILE_DEFINITIONS)
    separate_arguments(targetDefs)
    separate_arguments(PACXX_DEVICE_FLAGS)
    add_custom_command(
            OUTPUT ${outFile}
            COMMAND ${PACXX_COMPILER}
            ${PACXX_DEVICE_FLAGS}
            ${targetDefs}
            -isystem ${PACXX_INCLUDE_DIRECTORY}
            ${device_includes}
            -o ${outFile}
            -c ${srcFile}
            DEPENDS ${srcFile}
            IMPLICIT_DEPENDS CXX ${srcFile}
            WORKING_DIRECTORY ${binDir}
            COMMENT "Generating LLVM IR from ${srcFile}")

    set(bcName ${srcName}.bc_ir)

    add_custom_target(${bcName} DEPENDS ${outFile})

    add_dependencies(${targetName} ${bcName})

endfunction()

function(pacxx_embed_ir targetName bcFiles binDir)
    set(kernelName ${targetName}_kernel_ir)
    set(mspName ${targetName}_reflect_ir)
    set(outFile ${binDir}/kernel.bc)
    set(mspFile ${binDir}/reflect.bc)

    separate_arguments(PACXX_LINK_FLAGS)

    separate_arguments(bcFiles)

    add_custom_command(
            OUTPUT ${outFile}
            COMMAND ${PACXX_LINK} ${PACXX_LINK_FLAGS} ${bcFiles} -o ${outFile}
            COMMAND ${PACXX_OPT} -load=libPACXXTransforms.so -pacxx-codegen-prepare -inline ${outFile} -o ${outFile}
            WORKING_DIRECTORY ${binDir}
            DEPENDS ${bcFiles}
            COMMENT "Generating Kernel IR")

    add_custom_command(
            OUTPUT ${outFile}.o
            COMMAND ${PACXX_COMPILER} -DFILE='"${outFile}"' -DTAG='llvm' ${PACXX_ASM_WRAPPER} -c -o ${outFile}.o
            WORKING_DIRECTORY ${binDir}
            DEPENDS ${outFile}
            COMMENT "Preparing Kernel IR for linking")


    add_custom_target(${kernelName} DEPENDS ${outFile} ${outFile}.o)

    foreach (bcFile ${bcFiles})
        get_filename_component(srcName ${bcFile} NAME)
        add_dependencies(${kernelName} ${srcName}_ir)
    endforeach ()

    add_dependencies(${targetName} ${kernelName})

    target_link_libraries(${targetName} PUBLIC ${outFile}.o)
endfunction()

function(add_pacxx_libs_to_target targetName)
set_target_properties(${targetName} PROPERTIES LINK_FLAGS ${PACXX_LD_FLAGS})

target_link_libraries(${targetName} PUBLIC ${PACXX_RUNTIME_LIBRARY} PUBLIC PACXXTransforms PUBLIC ${PACXX_RV_LIBRARY}
            PUBLIC ${HIP_LINK_LIBRARIES} PUBLIC ${CUDA_LINK_LIBRARIES} PUBLIC ${PACXX_LLVM_LIBS} PUBLIC ${PACXX_LLVM_SYS_LIBS} PUBLIC ${TBB_LIBRARIES} PUBLIC ${PAPI_LIBRARIES})
endfunction(add_pacxx_libs_to_target)

function(add_pacxx_to_target targetName binDir srcFiles)
    get_target_property(ALREADY_A_PACXX_TARGET ${targetName} IS_PACXX_TARGET)
    if (NOT ALREADY_A_PACXX_TARGET EQUAL "1")
    set_target_properties(${targetName} PROPERTIES IS_PACXX_TARGET 1)
    set(bcFiles "")
    set(srcFiles ${srcFiles} ${ARGN})

    foreach (srcFile ${srcFiles})
        pacxx_generate_ir(${targetName} ${srcFile} ${binDir})
        get_filename_component(srcFilename ${srcFile} NAME)
        set(bcFiles "${bcFiles} ${binDir}/${srcFilename}.bc")
    endforeach ()

    pacxx_embed_ir(${targetName} ${bcFiles} ${binDir})
	add_pacxx_libs_to_target(${targetName})
    target_link_libraries(${targetName} PUBLIC libpacxx_main.a)


    set(PACXX_ADDITIONAL_LINKER_FLAGS "-Wl,-wrap=main")
    get_target_property(EXISTING_LINKGER_FLAGS ${targetName} LINK_FLAGS)
    set(NEW_LINK_FLAGS "${EXISTING_LINKGER_FLAGS} ${PACXX_ADDITIONAL_LINKER_FLAGS}")
    set_target_properties(${targetName} PROPERTIES LINK_FLAGS ${NEW_LINK_FLAGS})


    target_compile_options(${targetName} PUBLIC -Wno-ignored-attributes -Wno-attributes)
    endif()
endfunction(add_pacxx_to_target)
