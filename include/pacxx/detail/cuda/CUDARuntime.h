//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef PACXX_V2_CUDARUNTIME_H
#define PACXX_V2_CUDARUNTIME_H

#include "../IRRuntime.h"
#include "../msp/MSPEngine.h"
#include "CUDADeviceBuffer.h"
#include "CUDAKernel.h"
#include "PTXBackend.h"
#include "pacxx/detail/common/Exceptions.h"
#include <cstdlib>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <algorithm>

// forward declarations of cuda driver structs
// struct CUctx_st;
// typedef struct CUctx_st *CUcontext;
// struct CUmod_st;
// typedef struct CUmod_st *CUmodule;
// struct CUstream_st;
// typedef struct CUstream_st *cudaStream_t;
// enum cudaError;
// typedef enum cudaError cudaError_t;

#include <driver_types.h>
namespace pacxx {
namespace v2 {

class CUDARuntime : public IRRuntime {
public:
  using CompilerT = PTXBackend;

  static bool classof(const IRRuntime *rt) {
    return rt->getKind() == RuntimeKind::RK_CUDA;
  }

  static bool checkSupportedHardware();

  CUDARuntime(unsigned dev_id);

  virtual ~CUDARuntime();

  virtual void link(std::unique_ptr<llvm::Module> M) override;

  virtual Kernel &getKernel(const std::string &name) override;

  virtual size_t getPreferedMemoryAlignment() override;

  virtual size_t getPreferedVectorSize(size_t dtype_size) override;

  virtual size_t getPreferedVectorSizeInBytes() override;

  virtual size_t getConcurrentCores() override;

  virtual bool supportsUnifiedAddressing() override;

  template <typename T>
  DeviceBuffer<T> *allocateMemory(size_t count, T *host_ptr, MemAllocMode mode = Standard) {
    auto raw = std::make_unique<CUDARawDeviceBuffer>([this](CUDARawDeviceBuffer& buffer){ deleteRawMemory(&buffer); }, mode);
    raw->allocate(count * sizeof(T));
    auto wrapped = new DeviceBuffer<T>(std::move(raw));
    _memory.push_back(std::unique_ptr<DeviceBufferBase<void>>(
        reinterpret_cast<DeviceBufferBase<void> *>(wrapped)));
    if (host_ptr)
      wrapped->upload(host_ptr, count);
    return wrapped;
  }

  template <typename T> DeviceBuffer<T> *translateMemory(T *ptr) {
    auto It =
        std::find_if(_memory.begin(), _memory.end(), [&](const auto &element) {
          return reinterpret_cast<DeviceBuffer<T> *>(element.get())->get() == ptr;
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

  virtual RawDeviceBuffer *allocateRawMemory(size_t bytes, MemAllocMode mode = MemAllocMode::Standard) override;

  virtual void deleteRawMemory(RawDeviceBuffer *ptr) override;

  virtual void requestIRTransformation(Kernel &K) override;

  virtual const llvm::Module &getModule() override;

  virtual void synchronize() override;

private:
  void compileAndLink();

public:
  static void fireCallback(cudaStream_t stream, cudaError_t status,
                           void *userData) {
    (*reinterpret_cast<std::function<void()> *>(userData))();
  }

  CUcontext &getContext();

private:
  CUcontext _context;
  CUmodule _mod;
  std::unique_ptr<CompilerT> _compiler;
  std::map<std::string, std::unique_ptr<CUDAKernel>> _kernels;
  std::vector<cudaDeviceProp> _dev_props;

  struct callback_mem {
    size_t size;
    void *ptr;
  };

  std::list<callback_mem> _callbacks;
  bool _delayed_compilation;
};
}
}

#endif // PACXX_V2_CUDARUNTIME_H
