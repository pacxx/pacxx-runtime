//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <chrono>
#include "pacxx/detail/rocm/HIPEvent.h"

namespace pacxx{
namespace v2{

void HIPEvent::start() {
  _start = std::chrono::high_resolution_clock::now();
}

void HIPEvent::stop() {
  _end = std::chrono::high_resolution_clock::now();
}

double HIPEvent::result() {
  return std::chrono::duration_cast<std::chrono::microseconds>(_end - _start).count();
}

}
}