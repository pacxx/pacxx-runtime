//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "pacxx/detail/codegen/Reflection.h"
#include "pacxx/detail/codegen/Types.h"
#include "pacxx/detail/codegen/Kernel.h"
#include "pacxx/Executor.h"


extern "C" int __real_main(int argc, char* argv[]);

extern "C" int __wrap_main(int argc, char* argv[]);