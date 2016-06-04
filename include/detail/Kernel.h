//
// Created by mhaidl on 31/05/16.
//

#ifndef PACXX_V2_KERNEL_H
#define PACXX_V2_KERNEL_H

#include <vector>
#include "KernelConfiguration.h"

namespace pacxx {
namespace v2 {

class Kernel {
public:
  virtual ~Kernel() {};
  virtual void configurate(KernelConfiguration config) = 0;
  virtual void setArguments(const std::vector<char> &arg_buffer) = 0;
  virtual void launch() = 0;
};
}
}

#endif // PACXX_V2_KERNEL_H
