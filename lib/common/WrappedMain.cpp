//===-----------------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "pacxx/detail/common/TearDown.h"

extern "C" int __real_main(int argc, char* argv[]);

extern "C" int __wrap_main(int argc, char *argv[]) {
  int ret = __real_main(argc, argv);
  pacxx::v2::pacxxTearDown();
  return ret;
}