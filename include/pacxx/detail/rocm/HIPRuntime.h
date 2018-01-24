//
// Created by mhaidl on 30/05/16.
//

#ifndef PACXX_V2_HIPRUNTIME_H
#define PACXX_V2_HIPRUNTIME_H

#include "../Runtime.h"
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

class HIPRuntime : public Runtime {
public:
  using CompilerT = HSACOBackend;

  static bool classof(const Runtime *rt) {
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

  virtual std::unique_ptr<RawDeviceBuffer> allocateRawMemory(size_t bytes, MemAllocMode mode = MemAllocMode::Standard) override;

  //virtual void deleteRawMemory(RawDeviceBuffer *ptr) override;

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
