cmake_minimum_required(VERSION 3.5)

find_package(CUDA REQUIRED)

if (NOT PACXX_DIR)
    message(FATAL_ERROR
            "PACXX - Not found! (please set PACXX_DIR)")
else ()
    message(STATUS "PACXX - Found")
endif ()

set(PACXX_DIR ${PACXX_DIR} CACHE PATH "Path to PACXX")

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

# Set the path to llvm-config
find_program(PACXX_LLVM_CONFIG llvm-config PATHS
        ${PACXX_DIR} PATH_SUFFIXES bin)
if (EXISTS ${PACXX_LLVM_CONFIG})
    mark_as_advanced(PACXX_LLVM_CONFIG)
    message(STATUS "llvm-config (PACXX) - Found")
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
        ${PACXX_DIR} PATH_SUFFIXES bin)
if (EXISTS ${PACXX_COMPILER})
    mark_as_advanced(PACXX_COMPILER)
    message(STATUS "clang++ (PACXX) - Found")
else ()
    message(FATAL_ERROR "clang++ (PACXX) - Not found! (${PACXX_COMPILER})")
endif ()

find_program(PACXX_LINK llvm-link PATHS
        ${PACXX_DIR} PATH_SUFFIXES bin)
if (EXISTS ${PACXX_LINK})
    mark_as_advanced(PACXX_LINK)
    message(STATUS "llvm-link (PACXX) - Found")
else ()
    message(FATAL_ERROR "llvm-link (PACXX) - Not found! (${PACXX_LINK})")
endif ()

find_program(PACXX_OPT opt PATHS
        ${PACXX_DIR} PATH_SUFFIXES bin)
if (EXISTS ${PACXX_OPT})
    mark_as_advanced(PACXX_OPT)
    message(STATUS "opt (PACXX) - Found")
else ()
    message(FATAL_ERROR "opt (PACXX) - Not found! (${PACXX_OPT})")
endif ()


find_library(PACXX_RUNTIME_LIBRARY pacxxrt2 PATHS ${PACXX_DIR}
        HINTS ${PACXX_DIR}/lib PATH_SUFFIXES lib
        DOC "PACXX Runtime Library" NO_DEFAULT_PATH)

if (EXISTS ${PACXX_RUNTIME_LIBRARY})
    mark_as_advanced(PACXX_RUNTIME_LIBRARY)
    message(STATUS "libpacxxrt2.so - Found")
else ()
    message(FATAL_ERROR "libpacxxrt2.so - Not found!")
endif ()


find_file(PACXX_NVVM_DEVICE_BINDING nvvm_device_binding.bc PATHS ${PACXX_DIR}
        HINTS ${PACXX_DIR}/lib PATH_SUFFIXES lib
        DOC "PACXX Nvidia Device Binding" NO_DEFAULT_PATH)

if (EXISTS ${PACXX_NVVM_DEVICE_BINDING})
    mark_as_advanced(PACXX_NVVM_DEVICE_BINDING)
    message(STATUS "nvvm_device_binding.bc - Found")
else ()
    message(FATAL_ERROR "nvvm_device_binding.bc - Not found!")
endif ()

find_file(PACXX_ASM_WRAPPER embed_llvm.S PATHS ${PACXX_DIR}
        HINTS ${PACXX_DIR}/lib PATH_SUFFIXES lib
        DOC "PACXX Wrapper Assembly File" NO_DEFAULT_PATH)

if (EXISTS ${PACXX_ASM_WRAPPER})
    mark_as_advanced(PACXX_ASM_WRAPPER)
    message(STATUS "embed_llvm.S - Found")
else ()
    message(FATAL_ERROR "embed_llvm.S - Not found!")
endif ()

find_file(PACXX_ASM_WRAPPER_MSP embed_reflection.S PATHS ${PACXX_DIR}
        HINTS ${PACXX_DIR}/lib PATH_SUFFIXES lib
        DOC "PACXX Wrapper Assembly File (MSP)" NO_DEFAULT_PATH)

if (EXISTS ${PACXX_ASM_WRAPPER_MSP})
    mark_as_advanced(PACXX_ASM_WRAPPER_MSP)
    message(STATUS "embed_reflect.S - Found")
else ()
    message(FATAL_ERROR "embed_reflect.S - Not found!")
endif ()

set(PACXX_INCLUDE_DIRECTORY ${PACXX_DIR}/include)
if (NOT EXISTS ${PACXX_INCLUDE_DIRECTORY})
    message(FATAL_ERROR "PACXX includes - Not found!")
else ()
    message(STATUS "PACXX includes - Found")
endif ()


set(PACXX_DEVICE_FLAGS "-std=pacxx -O0 -S -emit-llvm -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -DNDEBUG -D__CUDA_DEVICE_CODE" CACHE "PACXX Device compilation flags" STRING)
set(PACXX_LINK_FLAGS "-suppress-warnings" CACHE "PACXX bytecode linker flags" STRING)
set(PACXX_OPT_FLAGS "-scalarizer -mem2reg -targetlibinfo -tbaa -basicaa -loop-rotate -simplifycfg -basiccg -functionattrs -sroa -domtree -early-cse -lazy-value-info -correlated-propagation -reassociate -domtree -loops -lcssa -loop-rotate -slsr -licm -loop-unswitch -loop-idiom -loop-deletion -loop-unroll -instsimplify -domtree -loops -lcssa -memdep -gvn -break-crit-edges -constmerge -pacxx_reflection -pacxx_inline -pacxx_dce -scalarizer -mem2reg -instcombine -simplifycfg -instcombine -pacxx_inline -pacxx_dce" CACHE "PACXX bytecode optimizer flags" STRING)
set(PACXX_OPT_FLAGS_MSP "-pacxx_reflection_cleaner -inline -instcombine" CACHE "PACXX bytecode optimizer flags (MSP)" STRING)


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

    separate_arguments(PACXX_DEVICE_FLAGS)
    add_custom_command(
            OUTPUT ${outFile}
            COMMAND ${PACXX_COMPILER}
            ${PACXX_DEVICE_FLAGS}
            -isystem ${PACXX_INCLUDE_DIRECTORY}
            ${device_includes}
            -o ${outFile}
            -c ${srcFile}
            DEPENDS ${WORKING_DIRECTORY}/${srcFile}
            IMPLICIT_DEPENDS CXX ${WORKING_DIRECTORY}/${srcFile}
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
    separate_arguments(PACXX_OPT_FLAGS)
    separate_arguments(PACXX_OPT_FLAGS_MSP)
    separate_arguments(bcFiles)

    add_custom_command(
            OUTPUT ${outFile}
            COMMAND ${PACXX_LINK} ${PACXX_LINK_FLAGS} ${bcFiles} -o ${outFile}
            COMMAND ${PACXX_LINK} ${PACXX_LINK_FLAGS} ${PACXX_NVVM_DEVICE_BINDING} ${outFile} -o ${outFile}
            COMMAND ${PACXX_OPT} ${PACXX_OPT_FLAGS} ${outFile} -o ${outFile}
            WORKING_DIRECTORY ${binDir}
            COMMENT "Generating Kernel IR")

    add_custom_command(
            OUTPUT ${outFile}.o
            COMMAND ${PACXX_COMPILER} -DFILE='"${outFile}"' ${PACXX_ASM_WRAPPER} -c -o ${outFile}.o
            WORKING_DIRECTORY ${binDir}
            COMMENT "Preparing Kernel IR for linking")

    add_custom_command(
            OUTPUT ${mspFile}
            COMMAND ${PACXX_OPT} ${PACXX_OPT_FLAGS_MSP} ${outFile} -o ${mspFile}
            COMMAND ${PACXX_COMPILER} -DFILE='"${mspFile}"' ${PACXX_ASM_WRAPPER_MSP} -c -o ${mspFile}.o
            WORKING_DIRECTORY ${binDir}
            COMMENT "Generating MSP IR")

    add_custom_command(
            OUTPUT ${mspFile}.o
            COMMAND ${PACXX_COMPILER} -DFILE='"${mspFile}"' ${PACXX_ASM_WRAPPER_MSP} -c -o ${mspFile}.o
            WORKING_DIRECTORY ${binDir}
            COMMENT "Preparing MSP IR for linking")


    add_custom_target(${kernelName} DEPENDS ${outFile} ${outFile}.o)
    add_custom_target(${mspName} DEPENDS ${mspFile} ${mspFile}.o)
    add_dependencies(${mspName} ${kernelName})

    foreach (bcFile ${bcFiles})
        get_filename_component(srcName ${bcFile} NAME)
        add_dependencies(${kernelName} ${srcName}_ir)
    endforeach ()

    add_dependencies(${targetName} ${mspName})
    add_dependencies(${targetName} ${kernelName})

    target_link_libraries(${targetName} PUBLIC ${outFile}.o PUBLIC ${mspFile}.o)
endfunction()

function(add_pacxx_to_target targetName binDir srcFiles)

    set(bcFiles "")
    set(srcFiles ${srcFiles} ${ARGN})

    foreach (srcFile ${srcFiles})
        pacxx_generate_ir(${targetName} ${srcFile} ${binDir})
        get_filename_component(srcFilename ${srcFile} NAME)
        set(bcFiles "${bcFiles} ${binDir}/${srcFilename}.bc")
    endforeach ()

    pacxx_embed_ir(${targetName} ${bcFiles} ${binDir})

    set_target_properties(${targetName} PROPERTIES LINK_FLAGS ${PACXX_LD_FLAGS})
    target_link_libraries(${targetName} PUBLIC ${PACXX_RUNTIME_LIBRARY}
            PUBLIC ${CUDA_LIBRARIES} PUBLIC cuda PUBLIC ${PACXX_LLVM_LIBS} PUBLIC ${PACXX_LLVM_SYS_LIBS})

    target_compile_options(${targetName} PUBLIC -Wno-ignored-attributes)

endfunction(add_pacxx_to_target)
