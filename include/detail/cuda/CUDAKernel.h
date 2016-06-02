//
// Created by mhaidl on 31/05/16.
//

#ifndef PACXX_V2_CUDAKERNEL_H
#define PACXX_V2_CUDAKERNEL_H

#include <cuda.h>
#include "detail/Kernel.h"
#include "detail/KernelConfiguration.h"

namespace pacxx
{
  namespace v2
  {
    class CUDAKernel : public Kernel
    {
      friend class CUDARuntime;
    private:
      CUDAKernel(CUfunction fptr);
    public:
      virtual~CUDAKernel();

      virtual void configurate(KernelConfiguration config) override;
      virtual void setArguments(const std::vector<char>& arg_buffer) override;
      virtual void launch() override;

    private:
      KernelConfiguration _config;
      std::vector<char> _args;
      size_t _args_size;
      std::vector<void*> _launch_args;
      CUfunction _fptr;
    };
  }
}

#endif //PACXX_V2_CUDAKERNEL_H
