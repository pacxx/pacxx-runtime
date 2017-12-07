//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "pacxx/detail/common/TearDown.h"
#include "pacxx/detail/common/ExecutorHelper.h"

extern const char llvm_start[];
extern const char llvm_end[];

extern "C" int __real_main(int argc, char* argv[]);

extern "C" int __wrap_main(int argc, char *argv[]) {
  pacxx::v2::registerModule(llvm_start, llvm_end);
  int ret = __real_main(argc, argv);
  pacxx::v2::pacxxTearDown();
  return ret;
}