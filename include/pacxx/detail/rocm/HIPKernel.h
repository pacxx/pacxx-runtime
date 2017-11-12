//
// Created by mhaidl on 31/05/16.
//

#ifndef PACXX_V2_HIPKERNEL_H
#define PACXX_V2_HIPKERNEL_H

#include "pacxx/detail/Kernel.h"
#include "pacxx/detail/KernelConfiguration.h"
#include <functional>
#include <map>
#include <string>
#define __HIP_PLATFORM_HCC__
#include <hip/hip_runtime.h>

namespace pacxx {
namespace v2 {
class HIPRuntime;

class HIPKernel : public Kernel {
  friend class HIPRuntime;

private:
  HIPKernel(HIPRuntime &runtime, hipFunction_t fptr, std::string name);

public:
  virtual ~HIPKernel();

  virtual void configurate(KernelConfiguration config) override;
  virtual void launch() override;


private:
  void overrideFptr(hipFunction_t fptr) { _fptr = fptr; }

private:
  HIPRuntime &_runtime;
  std::vector<void *> _launch_args;
  hipFunction_t _fptr;

};
}
}

#endif // PACXX_V2_HIPKERNEL_H
