//
// Created by mhaidl on 31/05/16.
//
#include "detail/KernelConfiguration.h"
#include <vector_types.h>

namespace pacxx {
namespace v2 {

Dimension3::Dimension3(size_t vx, size_t vy,  size_t vz)
    : x(vx), y(vy), z(vz) {}

Dimension3::Dimension3(dim3 px) : x(px.x), y(px.y), z(px.z) {}

dim3 Dimension3::getDim3() { return dim3(x, y, z); }

KernelConfiguration::KernelConfiguration(Dimension3 b, Dimension3 t,
                                         size_t sm)
    : blocks(b), threads(t), sm_size(sm) {}

KernelConfiguration::KernelConfiguration(size_t total_threads)
    : KernelConfiguration({DIV_UP(total_threads, NTHREADS)}, {NTHREADS}) {}

KernelConfiguration::KernelConfiguration()
    : KernelConfiguration(Dimension3(), Dimension3()) {}
}
}