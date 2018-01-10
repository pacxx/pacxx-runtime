//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "pacxx/detail/Kernel.h"
#include "pacxx/detail/KernelConfiguration.h"
#include <functional>
#include <map>
#include <string>

namespace pacxx {
namespace v2 {
class RemoteRuntime;

class RemoteKernel : public Kernel {
  friend class RemoteRuntime;

private:
  RemoteKernel(RemoteRuntime &runtime, std::string name);

public:
  virtual ~RemoteKernel();

  virtual void configurate(KernelConfiguration config) override;
  virtual void launch() override;

private:
  RemoteRuntime &_runtime;
};
}
}
