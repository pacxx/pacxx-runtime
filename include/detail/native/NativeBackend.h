//
// Created by mhaidl on 14/06/16.
//

#ifndef PACXX_V2_NATIVEBACKEND_H
#define PACXX_V2_NATIVEBACKEND_H

#include <llvm/Linker/Linker.h>
#include <llvm/IR/LegacyPassManager.h>

namespace pacxx
{
  namespace v2
  {
    class NativeBackend {
    public:
      NativeBackend();

      ~NativeBackend();

      llvm::legacy::PassManager& getPassManager();

      void linkInModule(std::unique_ptr<llvm::Module> M);

    private:
      llvm::legacy::PassManager _PM;
      std::unique_ptr<llvm::Module> _composite;
      llvm::Linker _linker;
    };
  }
}
#endif //PACXX_V2_NATIVEBACKEND_H
