//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef PACXX_V2_IRRUNTIME_H
#define PACXX_V2_IRRUNTIME_H

#include "DeviceBuffer.h"
#include "Kernel.h"
#include "pacxx/detail/msp/MSPEngine.h"
#include "pacxx/detail/common/Exceptions.h"
#include "pacxx/pacxx_config.h"
#include <list>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>

namespace llvm {
class Module;
}

namespace pacxx {
namespace v2 {

template <typename T> class Callback;

class CallbackBase;

class Runtime {
public:
  enum RuntimeKind { RK_CUDA, RK_Native, RK_HIP, RK_Remote };

private:
  const RuntimeKind _kind;

public:
  RuntimeKind getKind() const { return _kind; }

  Runtime(RuntimeKind kind);

  virtual ~Runtime();

  virtual void link(std::unique_ptr<llvm::Module> M) = 0;

  virtual Kernel &getKernel(const std::string &name) = 0;

  virtual size_t getPreferedMemoryAlignment() = 0;

  virtual size_t getPreferedVectorSize(size_t dtype_size) = 0;

  virtual size_t getPreferedVectorSizeInBytes() = 0;

  virtual size_t getConcurrentCores() = 0;

  virtual bool supportsUnifiedAddressing() = 0;

  virtual std::unique_ptr<RawDeviceBuffer> allocateRawMemory(size_t bytes,
                                             MemAllocMode mode) = 0;

  //virtual void deleteRawMemory(RawDeviceBuffer *ptr) = 0;

  virtual void initializeMSP(std::unique_ptr<llvm::Module> M);

  virtual void evaluateStagedFunctions(Kernel &K);

  virtual void requestIRTransformation(Kernel &K) = 0;

  virtual const llvm::Module &getModule();

  virtual void synchronize() = 0;

  virtual bool isSupportingDoublePrecission() { return true; }

  template <typename T>
  DeviceBuffer<T> *allocateMemory(size_t count,
                                  MemAllocMode mode = Standard) {
    auto raw = allocateRawMemory(count * sizeof(T), mode);
    auto wrapped = new DeviceBuffer<T>(std::move(raw));
    _memory.push_back(std::unique_ptr<DeviceBufferBase<void>>(
        reinterpret_cast<DeviceBufferBase<void> *>(wrapped)));
    return wrapped;
  }

  template <typename T> DeviceBuffer<T> *translateMemory(T *ptr) {
    auto It =
        std::find_if(_memory.begin(), _memory.end(), [&](const auto &element) {
          return reinterpret_cast<DeviceBuffer<T> *>(element.get())->get() ==
                 ptr;
        });

    if (It != _memory.end())
      return reinterpret_cast<DeviceBuffer<T> *>(It->get());
    else
      throw common::generic_exception(
          "supplied pointer not found in translation list");
  }

  template <typename T> void deleteMemory(DeviceBuffer<T> *ptr) {
    auto It =
        std::find_if(_memory.begin(), _memory.end(), [&](const auto &element) {
          return reinterpret_cast<DeviceBuffer<T> *>(element.get()) == ptr;
        });

    if (It != _memory.end())
      _memory.erase(It);
  }

protected:
  MSPEngine _msp_engine;
  std::unique_ptr<llvm::Module> _M, _rawM;
  std::list<std::unique_ptr<DeviceBufferBase<void>>> _memory;
};
} // namespace v2
} // namespace pacxx

#endif // PACXX_V2_IRRUNTIME_H
