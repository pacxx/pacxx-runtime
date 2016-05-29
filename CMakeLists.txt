cmake_minimum_required(VERSION 2.8)
project(PACXX_V2)

find_package(LLVM REQUIRED CONFIG)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

add_definitions(${LLVM_DEFINITIONS})

include_directories("./include")

add_library(pacxxrt2 ${SOURCE_FILES} include/Executor.h include/detail/IRCompiler.h src/PTXBackend.cpp include/detail/PTXBackend.h include/detail/Exceptions.h include/detail/Log.h include/detail/Common.h src/Common.cpp include/detail/CoreInitializer.h src/CoreInitializer.cpp)