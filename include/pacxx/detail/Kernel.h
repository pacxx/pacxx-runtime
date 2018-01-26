//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef PACXX_V2_KERNEL_H
#define PACXX_V2_KERNEL_H

#include "KernelConfiguration.h"
#include <vector>
#include <map>
#include <functional>

namespace pacxx {
namespace v2 {

class Runtime;

class Kernel {
public:
  Kernel(Runtime &runtime, std::string name);
  virtual ~Kernel(){};
  virtual void configurate(KernelConfiguration config) = 0;
  virtual KernelConfiguration getConfiguration() const;

  virtual void setStagedValue(int ref, long long value, bool inScope);
  virtual const std::map<int, long long> &getStagedValues() const;

  virtual void disableStaging();
  virtual bool requireStaging();
  virtual void setName(std::string name);
  virtual const std::string &getName() const;
  virtual void launch() = 0;

  virtual void profile() {
    return;
  }

  virtual void setCallback(std::function<void()> callback);

  virtual void setLambdaPtr(const void* ptr, size_t size) {
    _lambdaPtr = ptr;
    _lambdaSize = size;
  };

  virtual const void *getLambdaPtr() { return _lambdaPtr; };

protected:
  Runtime &_runtime_ref;
  KernelConfiguration _config;
  std::map<int, long long> _staged_values;
  bool _staged_values_changed;
  std::string _name;
  std::function<void()> _callback;
  bool _disable_staging;
  size_t _argBufferSize;
  const void *_lambdaPtr;
  size_t _lambdaSize;
};
}
}

#endif // PACXX_V2_KERNEL_H
