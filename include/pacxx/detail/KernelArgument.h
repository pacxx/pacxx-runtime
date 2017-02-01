//
// Created by mhaidl on 30/06/16.
//

#ifndef PACXX_V2_KERNELARGUMENTS_H
#define PACXX_V2_KERNELARGUMENTS_H

#include <cstddef>

namespace pacxx {
namespace v2 {
struct KernelArgument {
  KernelArgument(void *address, size_t size) : address(address), size(size) {}
  void *address;
  size_t size;
};
}
}
#endif // PACXX_V2_KERNELARGUMENTS_H
