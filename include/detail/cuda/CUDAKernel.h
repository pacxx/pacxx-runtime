//
// Created by mhaidl on 31/05/16.
//

#ifndef PACXX_V2_CUDAKERNEL_H
#define PACXX_V2_CUDAKERNEL_H

#include <cuda.h>
#include <map>
#include <string>
#include <functional>
#include "detail/Kernel.h"
#include "detail/KernelConfiguration.h"

namespace pacxx
{
  namespace v2
  {
    class CUDARuntime;

    class CUDAKernel : public Kernel
    {
      friend class CUDARuntime;
    private:
      CUDAKernel(CUDARuntime& runtime, CUfunction fptr);
    public:
      virtual~CUDAKernel();

      virtual void configurate(KernelConfiguration config) override;
      virtual KernelConfiguration getConfiguration() const override;
      virtual void setArguments(const std::vector<char>& arg_buffer) override;
      virtual const std::vector<char>& getArguments() const override;
      virtual void setHostArguments(const std::vector<char>& arg_buffer) override;
      virtual const std::vector<char>& getHostArguments() const override;
      virtual void setStagedValue(int ref, long long value) override;
      virtual const std::map<int, long long>& getStagedValues() const override;

      virtual void setName(std::string name) override;
      virtual const std::string& getName() const override;

      virtual void launch() override;

      virtual void setCallback(std::function<void()> callback) override { _callback = callback; };

    private:
      void overrideFptr(CUfunction fptr) { _fptr = fptr; }

    private:
      CUDARuntime& _runtime;
      KernelConfiguration _config;
      std::vector<char> _args;
      std::vector<char> _host_args;
      size_t _args_size;
      std::vector<void*> _launch_args;
      CUfunction _fptr;
      std::map<int, long long> _staged_values;
      bool _staged_values_changed;
      std::string _name;
      std::function<void()> _callback;
    };
  }
}

#endif //PACXX_V2_CUDAKERNEL_H
