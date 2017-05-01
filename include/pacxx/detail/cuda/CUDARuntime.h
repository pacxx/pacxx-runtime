//
// Created by mhaidl on 30/05/16.
//

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

// forward declarations of cuda driver structs
struct CUctx_st;
typedef struct CUctx_st *CUcontext;
struct CUmod_st;
typedef struct CUmod_st *CUmodule;

namespace pacxx {
namespace v2 {

class CUDARuntime : public IRRuntime {
public:
  using CompilerT = PTXBackend;

  static bool checkSupportedHardware();

  CUDARuntime(unsigned dev_id);

  virtual ~CUDARuntime();

  virtual RuntimeType getRuntimeType() override;

  virtual void link(std::unique_ptr<llvm::Module> M) override;

  virtual Kernel &getKernel(const std::string &name) override;

  virtual size_t getPreferedMemoryAlignment() override;

  virtual size_t getPreferedVectorSize(size_t dtype_size) override;

  virtual size_t getConcurrentCores() override;

  template <typename T>
  DeviceBuffer<T> *allocateMemory(size_t count, T *host_ptr) {
    CUDARawDeviceBuffer raw;
    raw.allocate(count * sizeof(T));
    auto wrapped = new CUDADeviceBuffer<T>(std::move(raw));
    _memory.push_back(std::unique_ptr<DeviceBufferBase>(
        static_cast<DeviceBufferBase *>(wrapped)));
    if (host_ptr)
      wrapped->upload(host_ptr, count);
    return wrapped;
  }

  template <typename T> DeviceBuffer<T> *translateMemory(T *ptr) {
    auto It =
        std::find_if(_memory.begin(), _memory.end(), [&](const auto &element) {
          return reinterpret_cast<CUDADeviceBuffer<T> *>(element.get())
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

  virtual RawDeviceBuffer *allocateRawMemory(size_t bytes) override;

  virtual void deleteRawMemory(RawDeviceBuffer *ptr) override;

  virtual void requestIRTransformation(Kernel &K) override;

  virtual const llvm::Module &getModule() override;

  virtual void synchronize() override;

  virtual llvm::legacy::PassManager &getPassManager() override;

private:
  void compileAndLink();

public:
  static void fireCallback(cudaStream_t stream, cudaError_t status,
                           void *userData) {
    (*reinterpret_cast<std::function<void()> *>(userData))();
  }

private:
  CUcontext _context;
  CUmodule _mod;
  std::unique_ptr<CompilerT> _compiler;
  std::map<std::string, std::unique_ptr<CUDAKernel>> _kernels;
  std::list<std::unique_ptr<DeviceBufferBase>> _memory;
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
