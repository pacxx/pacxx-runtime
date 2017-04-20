//
// Created by mhaidl on 31/05/16.
//
#include "pacxx/detail/KernelConfiguration.h"

namespace pacxx {
namespace v2 {

Dimension3::Dimension3(size_t vx, size_t vy, size_t vz) : x(vx), y(vy), z(vz) {}

KernelConfiguration::KernelConfiguration(Dimension3 b, Dimension3 t, unsigned executorID, size_t sm)
    : blocks(b), threads(t), executor(executorID),  sm_size(sm) {}

KernelConfiguration::KernelConfiguration(size_t total_threads)
    : KernelConfiguration({DIV_UP(total_threads, NTHREADS)}, {NTHREADS}, 0, 0) {}

KernelConfiguration::KernelConfiguration()
    : KernelConfiguration(Dimension3(), Dimension3(), 0, 0) {}
}
}