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
#include <algorithm>
#include <functional>
#include <map>
#include <numeric>
#include <string>

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
  void *_fptr;
  unsigned _runs;
};

} // namespace v2

} // namespace pacxx

#endif // PACXX_V2_NATIVEKERNEL_H
