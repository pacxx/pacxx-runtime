//
// Created by mhaidl on 31/05/16.
//

#ifndef PACXX_V2_CUDAKERNEL_H
#define PACXX_V2_CUDAKERNEL_H

#include "pacxx/detail/Kernel.h"
#include "pacxx/detail/KernelConfiguration.h"
#include <cuda.h>
#include <functional>
#include <map>
#include <string>

namespace pacxx {
namespace v2 {
class CUDARuntime;

class CUDAKernel : public Kernel {
  friend class CUDARuntime;

private:
  CUDAKernel(CUDARuntime &runtime, CUfunction fptr);

public:
  virtual ~CUDAKernel();

  virtual void configurate(KernelConfiguration config) override;
  virtual void setArguments(const std::vector<char> &arg_buffer) override;
  virtual void launch() override;


private:
  void overrideFptr(CUfunction fptr) { _fptr = fptr; }

private:
  CUDARuntime &_runtime;
  size_t _args_size;
  std::vector<void *> _launch_args;
  CUfunction _fptr;

};
}
}

#endif // PACXX_V2_CUDAKERNEL_H
