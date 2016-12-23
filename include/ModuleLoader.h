//
// Created by mhaidl on 29/05/16.
//

#ifndef PACXX_V2_MODULELOADER_H
#define PACXX_V2_MODULELOADER_H

#include <llvm/IR/LLVMContext.h>
#include <memory>

namespace llvm {
  class Module;
}

namespace pacxx {
  namespace v2 {

    class ModuleLoader
    {
    public:
      ModuleLoader(llvm::LLVMContext& ctx) : _ctx(ctx) {}
      std::unique_ptr<llvm::Module> loadIR(const std::string& IR);
      std::unique_ptr<llvm::Module> loadFile(const std::string& filename);
      std::unique_ptr<llvm::Module> loadInternal(const char* ptr, size_t size);

      std::unique_ptr<llvm::Module> loadAndLink(std::unique_ptr<llvm::Module> old, const std::string& filename);

      std::unique_ptr<llvm::Module> link(std::unique_ptr<llvm::Module> m1, std::unique_ptr<llvm::Module> m2);

    private:
      llvm::LLVMContext& _ctx;
    };

  }
}
#endif //PACXX_V2_MODULELOADER_H
