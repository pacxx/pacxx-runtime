//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef PACXX_V2_KERNELCONFIGURATION_H
#define PACXX_V2_KERNELCONFIGURATION_H

#include <cstddef>
#include <map>

#define DIV_UP(a, b) (((a) + (b)-1) / (b))
#define NTHREADS 128

namespace pacxx {
namespace v2 {

struct Dimension3 {
  Dimension3(size_t vx = 1, size_t vy = 1, size_t vz = 1);

  bool operator==(const Dimension3 &rhs) {
    return x == rhs.x && y == rhs.y && z == rhs.z;
  }

  bool operator!=(const Dimension3 &rhs) {
    return x != rhs.x || y != rhs.y || z != rhs.z;
  }

  std::map<std::string, std::size_t> ToStdContainer() const {
    return {
      {"x", x},
      {"y", y},
      {"z", z}
    };
  }

  size_t x, y, z;
};

struct KernelConfiguration {

  KernelConfiguration(Dimension3 b, Dimension3 t, size_t sm = 0);
  KernelConfiguration(size_t total_threads);
  KernelConfiguration();

  bool operator==(const KernelConfiguration &rhs) {
    return blocks == rhs.blocks && threads == rhs.threads;
  }

  bool operator!=(const KernelConfiguration &rhs) {
    return blocks != rhs.blocks || threads != rhs.threads;
  }

  std::pair<std::map<std::string, std::map<std::string, std::size_t>>, std::map<std::string, std::size_t>> ToStdContainer() const {
    return std::pair<std::map<std::string, std::map<std::string, std::size_t>>, std::map<std::string, std::size_t>> (
      {
        {"blocks", blocks.ToStdContainer()},
        {"threads", threads.ToStdContainer()}
      },
      {
        {"shmem", sm_size}
      }
    );
  }

  Dimension3 blocks;
  Dimension3 threads;
  size_t sm_size;
  unsigned executor;
};
}
}
#endif // PACXX_V2_KERNELCONFIGURATION_H
