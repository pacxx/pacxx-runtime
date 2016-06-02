//
// Created by mhaidl on 31/05/16.
//

#ifndef PACXX_V2_KERNELCONFIGURATION_H
#define PACXX_V2_KERNELCONFIGURATION_H

#include <cstddef>
#include <vector_types.h>


#define DIV_UP(a, b) (((a) + (b)-1) / (b))
#define NTHREADS 128

namespace pacxx {
namespace v2 {

struct Dimension3 {
  Dimension3(size_t vx = 1, size_t vy = 1, size_t vz = 1);
  Dimension3(dim3 px);
  dim3 getDim3();
  size_t x, y, z;
};

struct KernelConfiguration {

  KernelConfiguration(Dimension3 b, Dimension3 t, size_t sm = 0);
  KernelConfiguration(size_t total_threads);
  KernelConfiguration();

  Dimension3 blocks;
  Dimension3 threads;
  size_t sm_size;
};
}
}
#endif // PACXX_V2_KERNELCONFIGURATION_H
