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

#include "pacxx/pacxx_config.h"
#include "DeviceBuffer.h"
#include "Kernel.h"
#include <string>
#include <vector>
#include "pacxx/detail/msp/MSPEngine.h"

namespace llvm {
	class Module;
}

namespace pacxx {
namespace v2 {

template <typename T> class Callback;

class CallbackBase;

class IRRuntime {
public:
  enum RuntimeKind {
    RK_CUDA,
    RK_Native, 
    RK_HIP
  };
private:
  const RuntimeKind _kind;
public:
  RuntimeKind getKind() const { return _kind; }

  IRRuntime(RuntimeKind kind);

  virtual ~IRRuntime(){};

  virtual void link(std::unique_ptr<llvm::Module> M) = 0;

  virtual Kernel &getKernel(const std::string &name) = 0;

  virtual size_t getPreferedMemoryAlignment() = 0;

  virtual size_t getPreferedVectorSize(size_t dtype_size) = 0;

  virtual size_t getPreferedVectorSizeInBytes() = 0;

  virtual size_t getConcurrentCores() = 0;

  virtual bool supportsUnifiedAddressing() = 0;

  virtual RawDeviceBuffer *allocateRawMemory(size_t bytes, MemAllocMode mode) = 0;

  virtual void deleteRawMemory(RawDeviceBuffer *ptr) = 0;

  virtual void initializeMSP(std::unique_ptr<llvm::Module> M);

  virtual void evaluateStagedFunctions(Kernel &K);

  virtual void requestIRTransformation(Kernel &K) = 0;

  virtual const llvm::Module &getModule();

  virtual void synchronize() = 0;

  virtual bool isSupportingDoublePrecission(){ return true; }

protected:
  MSPEngine _msp_engine;
  std::unique_ptr<llvm::Module> _M, _rawM;
};
}
}

#endif // PACXX_V2_IRRUNTIME_H
