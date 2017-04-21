//
// Created by mhaidl on 31/05/16.
//

#ifndef PACXX_V2_KERNEL_H
#define PACXX_V2_KERNEL_H

#include "KernelConfiguration.h"
#include <vector>
#include <map>
#include <functional>

namespace pacxx {
namespace v2 {

class Kernel {
public:
  virtual ~Kernel(){};
  virtual void configurate(KernelConfiguration config) = 0;
  virtual KernelConfiguration getConfiguration() const = 0;
  virtual void setArguments(const std::vector<char> &arg_buffer) = 0;
  virtual const std::vector<char> &getArguments() const = 0;
  virtual const std::vector<size_t> &getArugmentBufferOffsets() = 0;
  virtual size_t getArgBufferSize() = 0;

  virtual size_t getHostArgumentsSize() const = 0;

  virtual void setHostArgumentsSize(size_t size) = 0;
  virtual void setHostArguments(const std::vector<char> &arg_buffer) = 0;
  virtual const std::vector<char> &getHostArguments() const = 0;

  virtual void setStagedValue(int ref, long long value, bool inScope) = 0;
  virtual const std::map<int, long long> &getStagedValues() const = 0;

  virtual void disableStaging() = 0;

  virtual bool requireStaging() = 0;
  virtual void setName(std::string name) = 0;
  virtual const std::string &getName() const = 0;
  virtual void launch() = 0;

  virtual void setCallback(std::function<void()> callback) = 0;
};
}
}

#endif // PACXX_V2_KERNEL_H
