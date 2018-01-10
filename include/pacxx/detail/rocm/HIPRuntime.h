//
// Created by mhaidl on 30/05/16.
//

#ifndef PACXX_V2_HIPRUNTIME_H
#define PACXX_V2_HIPRUNTIME_H

#include "../IRRuntime.h"
#include "../msp/MSPEngine.h"
#include "HIPDeviceBuffer.h"
#include "HIPKernel.h"
#include "HSACOBackend.h"
#include "pacxx/detail/common/Exceptions.h"
#include <cstdlib>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <algorithm>

// forward declarations of cuda driver structs
struct ihipCtx_t;
struct ihipModule_t;
struct ihipStream_t;
struct hipDeviceProp_t;
#ifdef __HIP_PLATFORM_HCC__
typedef struct ihipCtx_t *hipCtx_t;
typedef struct ihipModule_t *hipModule_t;
typedef struct ihipStream_t *hipStream_t;
#else 
struct CUctx_st;
typedef struct CUctx_st *CUcontext;
struct CUmod_st;
typedef struct CUmod_st *CUmodule;
struct CUstream_st;
typedef struct CUstream_st *cudaStream_t;

typedef CUcontext hipCtx_t;
typedef CUmodule hipModule_t; 
typedef cudaStream_t hipStream_t; 
#endif 

namespace pacxx {
namespace v2 {

class HIPRuntime : public IRRuntime {
public:
  using CompilerT = HSACOBackend;

  static bool classof(const IRRuntime *rt) {
    return rt->getKind() == RuntimeKind::RK_HIP;
  }

  static bool checkSupportedHardware();

  HIPRuntime(unsigned dev_id);

  virtual ~HIPRuntime();

  virtual void link(std::unique_ptr<llvm::Module> M) override;

  virtual Kernel &getKernel(const std::string &name) override;

  virtual size_t getPreferedMemoryAlignment() override;

  virtual size_t getPreferedVectorSize(size_t dtype_size) override;

  virtual size_t getPreferedVectorSizeInBytes() override;

  virtual size_t getConcurrentCores() override;

  virtual bool supportsUnifiedAddressing() override;

  template <typename T>
  DeviceBuffer<T> *allocateMemory(size_t count, T *host_ptr, MemAllocMode mode = Standard) {
    auto raw = std::make_unique<HIPRawDeviceBuffer>([this](HIPRawDeviceBuffer& buffer){ deleteRawMemory(&buffer); }, mode);
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
          return reinterpret_cast<DeviceBuffer<T> *>(element.get())
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

  virtual RawDeviceBuffer *allocateRawMemory(size_t bytes, MemAllocMode mode = MemAllocMode::Standard) override;

  virtual void deleteRawMemory(RawDeviceBuffer *ptr) override;

  virtual void requestIRTransformation(Kernel &K) override;

  virtual const llvm::Module &getModule() override;

  virtual void synchronize() override;


private:
  void compileAndLink();

public:
  static void fireCallback(hipStream_t stream, int status,
                           void *userData) {
    (*reinterpret_cast<std::function<void()> *>(userData))();
  }

private:
  hipCtx_t _context;
  hipModule_t _mod;
  std::unique_ptr<CompilerT> _compiler;
  std::map<std::string, std::unique_ptr<HIPKernel>> _kernels;
  std::vector<hipDeviceProp_t> _dev_props;

  struct callback_mem {
    size_t size;
    void *ptr;
  };

  std::list<callback_mem> _callbacks;
  bool _delayed_compilation;
};
}
}

#endif // PACXX_V2_HIPRUNTIME_H
