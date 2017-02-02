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
  virtual KernelConfiguration getConfiguration() const override;
  virtual void setArguments(const std::vector<char> &arg_buffer) override;
  virtual const std::vector<char> &getArguments() const override;
  virtual void setHostArguments(const std::vector<char> &arg_buffer) override;
  virtual const std::vector<char> &getHostArguments() const override;

  virtual void setStagedValue(int ref, long long value, bool inScope) override;

  virtual void disableStaging() override;

  virtual size_t getHostArgumentsSize() const override;
  virtual void setHostArgumentsSize(size_t size) override;

  virtual bool requireStaging() override;
  virtual const std::map<int, long long> &getStagedValues() const override;

  virtual void setName(std::string name) override;
  virtual const std::string &getName() const override;

  virtual void launch() override;

  virtual void setCallback(std::function<void()> callback) override {
    _callback = callback;
  };

private:
  void overrideFptr(void *fptr) { _fptr = fptr; }

private:
  NativeRuntime &_runtime;
  KernelConfiguration _config;
  std::vector<char> _args;
  std::vector<char> _host_args;
  void *_fptr;
  std::map<int, long long> _staged_values;
  bool _staged_values_changed;
  std::string _name;
  std::function<void()> _callback;
  bool _disable_staging;
  size_t _hostArgBufferSize;
};

} // v2 namespace

} // pacxx namespace

#endif // PACXX_V2_NATIVEKERNEL_H
