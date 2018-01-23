//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "pacxx/detail/Kernel.h"
#include "pacxx/detail/IRRuntime.h"

namespace pacxx {
namespace v2 {

Kernel* IRProfiler::_kernel = nullptr;

IRProfiler::IRProfiler() {
  InFilePath = common::GetEnv("PACXX_PROF_IN");
  if (InFilePath.empty()) InFilePath = "PACXX.prof";
  OutFilePath = common::GetEnv("PACXX_PROF_OUT");
  if (std::getenv("PACXX_NO_PROF")) _enabled = false;
  else _enabled = true;
}

bool IRProfiler::enabled() {
	return _enabled;
}

void IRProfiler::updateKernel(Kernel *kernel) {
	_kernel = kernel;
}

}
}
