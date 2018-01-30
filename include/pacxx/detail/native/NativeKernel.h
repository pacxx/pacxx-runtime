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
  virtual void profile() override;

  NativeRuntime &getRuntime();

private:
  void overrideFptr(void *fptr) { _fptr = fptr; }

private:
  NativeRuntime &_runtime;
  void *_fptr;
  unsigned _runs;
};

} // v2 namespace

} // pacxx namespace

#endif // PACXX_V2_NATIVEKERNEL_H
