//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "pacxx/Executor.h"
#include "pacxx/detail/common/Common.h"

namespace pacxx{
namespace v2 {

void pacxxTearDown(){
  auto &executors = Executor::getExecutors();
  delete &executors;
}
}
}
