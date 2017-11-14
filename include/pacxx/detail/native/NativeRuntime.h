//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef PACXX_V2_NATIVERUNTIME_H
#define PACXX_V2_NATIVERUNTIME_H

#include "../IRRuntime.h"
#include "NativeBackend.h"
#include "NativeDeviceBuffer.h"
#include "NativeKernel.h"
#include "pacxx/detail/common/Exceptions.h"
#include "pacxx/detail/msp/MSPEngine.h"
#include <list>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <map>
#include <memory>
#include <string>
#include <thread>

namespace pacxx {
namespace v2 {

class NativeRuntime : public IRRuntime {
public:
  using CompilerT = NativeBackend;

  static bool classof(const IRRuntime *rt) {
    return rt->getKind() == RuntimeKind::RK_Native;
  }

  NativeRuntime(unsigned dev_id);
  virtual ~NativeRuntime();

  virtual void link(std::unique_ptr<llvm::Module> M) override;

  virtual Kernel &getKernel(const std::string &name) override;

  virtual size_t getPreferedMemoryAlignment() override;

  virtual size_t getPreferedVectorSize(size_t dtype_size) override;

  virtual size_t getPreferedVectorSizeInBytes() override;

  virtual size_t getConcurrentCores() override;

  virtual bool supportsUnifiedAddressing() override;

  template <typename T>
  DeviceBuffer<T> *allocateMemory(size_t count, T *host_ptr, MemAllocMode mode = Standard) {
    NativeRawDeviceBuffer rawBuffer([this](NativeRawDeviceBuffer& buffer){ deleteRawMemory(&buffer); });

    auto bytes = count * sizeof(T);

    if (host_ptr)
      rawBuffer.allocate(bytes, reinterpret_cast<char *>(host_ptr));
    else
      rawBuffer.allocate(bytes, getPreferedVectorSizeInBytes());

    auto wrapped = new NativeDeviceBuffer<T>(std::move(rawBuffer));
    _memory.push_back(std::unique_ptr<DeviceBufferBase>(
        static_cast<DeviceBufferBase *>(wrapped)));
    return wrapped;
  }

  template <typename T> DeviceBuffer<T> *translateMemory(T *ptr) {
    auto It =
        std::find_if(_memory.begin(), _memory.end(), [&](const auto &element) {
          return reinterpret_cast<NativeDeviceBuffer<T> *>(element.get())
                     ->get() == ptr;
        });

    if (It != _memory.end())
      return reinterpret_cast<DeviceBuffer<T> *>(It->get());
    else
      throw common::generic_exception(
          "supplied pointer not found in translation list");
  }

  template <typename T> void deleteMemory(DeviceBuffer<T> *ptr) {
    auto It =
        std::find_if(_memory.begin(), _memory.end(),
                     [&](const auto &element) { return element.get() == ptr; });

    if (It != _memory.end())
      _memory.erase(It);
  }

  virtual RawDeviceBuffer *allocateRawMemory(size_t bytes, MemAllocMode mode = Standard) override;

  virtual void deleteRawMemory(RawDeviceBuffer *ptr) override;

  virtual void requestIRTransformation(Kernel &K) override;

  virtual void synchronize() override;

private:
  void compileAndLink();

private:
  llvm::Module *_CPUMod;

  std::unique_ptr<CompilerT> _compiler;
  std::map<std::string, std::unique_ptr<NativeKernel>> _kernels;
  std::list<std::unique_ptr<DeviceBufferBase>> _memory;
  llvm::StringMap<bool> _host_features;
  bool _delayed_compilation;

};
}
}

#endif // PACXX_V2_NATIVERUNTIME_H
