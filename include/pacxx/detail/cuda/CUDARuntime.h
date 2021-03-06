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

#include "../Runtime.h"
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

class CUDARuntime : public Runtime {
public:
  using CompilerT = PTXBackend;

  static bool classof(const Runtime *rt) {
    return rt->getKind() == RuntimeKind::RK_CUDA;
  }

  virtual bool checkSupportedHardware() override;

  CUDARuntime(unsigned dev_id);

  virtual ~CUDARuntime();

  virtual void link(std::unique_ptr<llvm::Module> M) override;

  virtual Kernel &getKernel(const std::string &name) override;

  virtual size_t getPreferedMemoryAlignment() override;

  virtual size_t getPreferedVectorSize(size_t dtype_size) override;

  virtual size_t getPreferedVectorSizeInBytes() override;

  virtual size_t getConcurrentCores() override;

  virtual bool supportsUnifiedAddressing() override;

  virtual std::unique_ptr<RawDeviceBuffer> allocateRawMemory(size_t bytes, MemAllocMode mode = MemAllocMode::Standard) override;

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
