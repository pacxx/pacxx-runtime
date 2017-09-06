//
// Created by mhaidl on 04/06/16.
//

#pragma once

#include "pacxx/detail/codegen/Reflection.h"
#include "pacxx/detail/codegen/Types.h"
#include "pacxx/detail/codegen/Kernel.h"
#include "pacxx/Executor.h"


extern "C" int __real_main(int argc, char* argv[]);

extern "C" int __wrap_main(int argc, char* argv[]);