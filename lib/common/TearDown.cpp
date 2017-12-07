//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "pacxx/detail/common/ExecutorHelper.h"
#include "pacxx/Executor.h"
#include <llvm/IR/Module.h>

namespace pacxx {
namespace v2 {

void pacxxTearDown() {
  auto executors = getExecutorMemory();
  delete executors;
}
} // namespace v2
} // namespace pacxx
