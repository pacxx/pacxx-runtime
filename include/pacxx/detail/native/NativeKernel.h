//
// Created by lars on 07/10/16.
//

#ifndef PACXX_V2_NATIVEKERNEL_H
#define PACXX_V2_NATIVEKERNEL_H

#include "pacxx/detail/Kernel.h"
#include "pacxx/detail/KernelConfiguration.h"
#include <functional>
#include <llvm/ExecutionEngine/GenericValue.h>
#include <llvm/IR/Function.h>
#include <map>
#include <string>

namespace pacxx {

namespace v2 {

class NativeRuntime;

class NativeKernel : public Kernel {

  friend class NativeRuntime;

private:
  NativeKernel(NativeRuntime &runtime, void *fptr);

public:
  virtual ~NativeKernel();

  virtual void configurate(KernelConfiguration config) override;
  virtual void launch() override;

private:
  void overrideFptr(void *fptr) { _fptr = fptr; }

private:
  NativeRuntime &_runtime;
  void *_fptr;

};

// Get the median of an unordered set of numbers of arbitrary
// type (this will modify the underlying dataset).
template <typename It>
typename std::iterator_traits<It>::value_type median(It begin, It end)
{
    auto size = std::distance(begin, end);
    std::nth_element(begin, begin + size / 2, end);
    return *std::next(begin, size / 2);
}

} // v2 namespace

} // pacxx namespace

#endif // PACXX_V2_NATIVEKERNEL_H
