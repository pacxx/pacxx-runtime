//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef PACXX_V2_NATIVEKERNEL_H
#define PACXX_V2_NATIVEKERNEL_H

#include "pacxx/detail/Kernel.h"
#include "pacxx/detail/KernelConfiguration.h"
#include <functional>
#include <llvm/ExecutionEngine/GenericValue.h>
#include <llvm/IR/Function.h>
#include <map>
#include <string>
#include <numeric>
#include <algorithm>

namespace pacxx {

namespace v2 {

class NativeRuntime;

class NativeKernel : public Kernel {

  friend class NativeRuntime;

private:
  NativeKernel(NativeRuntime &runtime, void *fptr, std::string name);

public:
  virtual ~NativeKernel();

  virtual void configurate(KernelConfiguration config) override;
  virtual void launch() override;

private:
  void overrideFptr(void *fptr) { _fptr = fptr; }

private:
  NativeRuntime &_runtime;
  void *_fptr;
  unsigned _runs;
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

template<typename It>
typename std::iterator_traits<It>::value_type average(It begin, It end) {
  auto size = std::distance(begin, end);
  auto sum = std::accumulate(begin, end, 0);
  return sum / size;
}

template<typename It>
typename std::iterator_traits<It>::value_type deviation(It begin, It end) {
  auto size = std::distance(begin, end);
  auto avg = average(begin, end);

  return sqrt(std::accumulate(begin, end, 0, [=](auto sum, auto val) {
    return sum + (val - avg) * (val - avg);
  }) / size);
}


} // v2 namespace

} // pacxx namespace

#endif // PACXX_V2_NATIVEKERNEL_H
