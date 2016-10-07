//
// Created by mhaidl on 14/06/16.
//

#ifndef PACXX_V2_NATIVERUNTIME_H
#define PACXX_V2_NATIVERUNTIME_H

#include <string>
#include <memory>
#include <map>
#include <list>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include "../IRRuntime.h"
#include "NativeBackend.h"
#include "NativeKernel.h"

namespace pacxx
{
  namespace v2
  {
    class NativeRuntime : public IRRuntime<NativeRuntime>
    {
    public:

      using CompilerT = NativeBackend;

      NativeRuntime(unsigned dev_id);
      virtual ~NativeRuntime();

      virtual void link(std::unique_ptr<llvm::Module> M) override;

      virtual Kernel& getKernel(const std::string& name) override;

      virtual size_t getPreferedMemoryAlignment() override;

      template <typename T>
      DeviceBuffer<T>* allocateMemory(size_t count) {
        return nullptr;
      }

      virtual RawDeviceBuffer* allocateRawMemory(size_t bytes) override;

      virtual void deleteRawMemory(RawDeviceBuffer* ptr) override;

      virtual void initializeMSP(std::unique_ptr <llvm::Module> M) override;

      virtual void evaluateStagedFunctions(Kernel& K) override;

      virtual void requestIRTransformation(Kernel& K) override;

      virtual const llvm::Module& getModule() override;

      virtual void synchronize() override;

      virtual llvm::legacy::PassManager& getPassManager() override;

    private:
      llvm::Module* _CPUMod;
      std::unique_ptr<llvm::Module> _M;
      std::unique_ptr<CompilerT> _compiler;
      std::map<std::string, std::unique_ptr<NativeKernel>> _kernels;
    };
  }
}

#endif //PACXX_V2_NATIVERUNTIME_H
