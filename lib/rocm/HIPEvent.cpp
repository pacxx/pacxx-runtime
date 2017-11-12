//
// Created by m_haid02 on 29.06.17.
//

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