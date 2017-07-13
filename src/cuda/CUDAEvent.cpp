//
// Created by m_haid02 on 29.06.17.
//

#include <chrono>
#include "pacxx/detail/cuda/CUDAEvent.h"

namespace pacxx{
namespace v2{

void CUDAEvent::start() {
  _start = std::chrono::high_resolution_clock::now();
}

void CUDAEvent::stop() {
  _end = std::chrono::high_resolution_clock::now();
}

double CUDAEvent::result() {
  return std::chrono::duration_cast<std::chrono::microseconds>(_end - _start).count();
}

}
}