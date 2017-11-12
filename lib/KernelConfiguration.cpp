//===-----------------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "pacxx/detail/KernelConfiguration.h"

namespace pacxx {
namespace v2 {

Dimension3::Dimension3(size_t vx, size_t vy, size_t vz) : x(vx), y(vy), z(vz) {}

KernelConfiguration::KernelConfiguration(Dimension3 b, Dimension3 t,  size_t sm)
    : blocks(b), threads(t),  sm_size(sm) {}

KernelConfiguration::KernelConfiguration(size_t total_threads)
    : KernelConfiguration({DIV_UP(total_threads, NTHREADS)}, {NTHREADS}, 0) {}

KernelConfiguration::KernelConfiguration()
    : KernelConfiguration(Dimension3(), Dimension3(), 0) {}
}
}