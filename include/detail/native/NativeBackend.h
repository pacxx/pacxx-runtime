//
// Created by mhaidl on 14/06/16.
//

#ifndef PACXX_V2_NATIVEBACKEND_H
#define PACXX_V2_NATIVEBACKEND_H

#include <llvm/Linker/Linker.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>

namespace pacxx
{
  namespace v2
  {
    class NativeBackend {
    public:
      NativeBackend();

      ~NativeBackend();

      llvm::legacy::PassManager& getPassManager();

      llvm::Module* compile(llvm::Module& M);

      void* getFunctionPtr(llvm::Module* module, const std::string name);

      static std::unique_ptr<llvm::Module> createModule(llvm::LLVMContext &Context, const std::string IR);

    private:
      void linkInModule(llvm::Module& M);
      void applyPasses(llvm::Module& M);

    private:
      llvm::legacy::PassManager _PM;
      const llvm::Target *_target;
      std::unique_ptr<llvm::Module> _composite;
      llvm::Linker _linker;
      llvm::ExecutionEngine *_JITEngine;
      bool _pmInitialized;

    };
  }
}
#endif //PACXX_V2_NATIVEBACKEND_H
