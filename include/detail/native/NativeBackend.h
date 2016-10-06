//
// Created by mhaidl on 14/06/16.
//

#ifndef PACXX_V2_NATIVEBACKEND_H
#define PACXX_V2_NATIVEBACKEND_H

#include <llvm/Linker/Linker.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/TargetRegistry.h>

namespace pacxx
{
  namespace v2
  {
    class NativeBackend {
    public:
      NativeBackend();

      ~NativeBackend();

      llvm::legacy::PassManager& getPassManager();

      void compile(llvm::Module& M);

      static std::unique_ptr<llvm::Module> createModule(llvm::LLVMContext &Context);

    private:
      void linkInModule(llvm::Module& M);
      void applyPasses(llvm::Module& M);

    private:
      llvm::legacy::PassManager _PM;
      const llvm::Target *_target;
      std::unique_ptr<llvm::Module> _composite;
      llvm::Linker _linker;
      bool _pmInitialized;

    };
  }
}
#endif //PACXX_V2_NATIVEBACKEND_H
