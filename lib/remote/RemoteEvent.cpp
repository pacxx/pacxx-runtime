//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <chrono>
#include "pacxx/detail/remote/RemoteEvent.h"

namespace pacxx{
namespace v2{

void RemoteEvent::start() {
  _start = std::chrono::high_resolution_clock::now();
}

void RemoteEvent::stop() {
  _end = std::chrono::high_resolution_clock::now();
}

double RemoteEvent::result() {
  return std::chrono::duration_cast<std::chrono::microseconds>(_end - _start).count();
}

}
}