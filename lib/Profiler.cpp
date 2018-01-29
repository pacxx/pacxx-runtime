//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "pacxx/detail/Kernel.h"
#include "pacxx/detail/Runtime.h"

namespace pacxx {
namespace v2 {

Kernel* Profiler::_kernel = nullptr;

Profiler::Profiler() {
  InFilePath = common::GetEnv("PACXX_PROF_IN");
  if (InFilePath.empty()) InFilePath = "PACXX.prof";
  OutFilePath = common::GetEnv("PACXX_PROF_OUT");
  if (std::getenv("PACXX_NO_PROF")) _enabled = false;
  else _enabled = true;
}

bool Profiler::enabled() {
	return _enabled;
}

void Profiler::updateKernel(Kernel *kernel) {
	_kernel = kernel;
}

}
}
