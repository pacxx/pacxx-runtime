//
// Created by mhaidl on 30/05/16.
//

#ifndef PACXX_V2_IRRUNTIME_H
#define PACXX_V2_IRRUNTIME_H

#include "DeviceBuffer.h"
#include "Kernel.h"
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <string>
#include <vector>

namespace pacxx {
namespace v2 {
template <typename T> class Callback;

class CallbackBase;

enum RuntimeType
{
  CUDARuntimeTy,
  NativeRuntimeTy
};

class IRRuntime {
public:
  virtual ~IRRuntime(){};

  virtual RuntimeType getRuntimeType() = 0;

  virtual void link(std::unique_ptr<llvm::Module> M) = 0;

  virtual Kernel &getKernel(const std::string &name) = 0;

  virtual size_t getPreferedMemoryAlignment() = 0;

  virtual size_t getPreferedVectorSize(size_t dtype_size) = 0;

  virtual size_t getConcurrentCores() = 0;

  virtual RawDeviceBuffer *allocateRawMemory(size_t bytes) = 0;

  virtual void deleteRawMemory(RawDeviceBuffer *ptr) = 0;

  virtual void initializeMSP(std::unique_ptr<llvm::Module> M) = 0;

  virtual void evaluateStagedFunctions(Kernel &K) = 0;

  virtual void requestIRTransformation(Kernel &K) = 0;

  virtual const llvm::Module &getModule() = 0;

  virtual void synchronize() = 0;

  virtual llvm::legacy::PassManager &getPassManager() = 0;

};
}
}

#endif // PACXX_V2_IRRUNTIME_H
