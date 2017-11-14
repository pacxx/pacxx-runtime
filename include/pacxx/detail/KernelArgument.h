//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

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
