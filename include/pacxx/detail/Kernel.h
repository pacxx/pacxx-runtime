//
// Created by mhaidl on 31/05/16.
//

#ifndef PACXX_V2_KERNEL_H
#define PACXX_V2_KERNEL_H

#include "KernelConfiguration.h"
#include <vector>
#include <map>
#include <functional>

namespace pacxx {
namespace v2 {

class IRRuntime;

class Kernel {
public:
  Kernel(IRRuntime &runtime);
  virtual ~Kernel(){};
  virtual void configurate(KernelConfiguration config) = 0;
  virtual KernelConfiguration getConfiguration() const;
  virtual void setArguments(const std::vector<char> &arg_buffer);
  virtual const std::vector<char> &getArguments() const;
  virtual const std::vector<size_t> &getArugmentBufferOffsets();
  virtual size_t getArgBufferSize();

  virtual size_t getHostArgumentsSize() const;

  virtual void setHostArgumentsSize(size_t size);
  virtual void setHostArguments(const std::vector<char> &arg_buffer);
  virtual const std::vector<char> &getHostArguments() const;

  virtual void setStagedValue(int ref, long long value, bool inScope);
  virtual const std::map<int, long long> &getStagedValues() const;

  virtual void disableStaging();

  virtual bool requireStaging();
  virtual void setName(std::string name);
  virtual const std::string &getName() const;
  virtual void launch() = 0;

  virtual void setCallback(std::function<void()> callback);

protected:
  IRRuntime &_runtime_ref;
  KernelConfiguration _config;
  std::vector<char> _args;
  std::vector<size_t> _arg_offsets;
  std::vector<char> _host_args;
  std::map<int, long long> _staged_values;
  bool _staged_values_changed;
  std::string _name;
  std::function<void()> _callback;
  bool _disable_staging;
  size_t _argBufferSize;
  size_t _hostArgBufferSize;
};
}
}

#endif // PACXX_V2_KERNEL_H
