//
// Created by mhaidl on 14/06/16.
//

#ifndef PACXX_V2_NATIVEBACKEND_H
#define PACXX_V2_NATIVEBACKEND_H

#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/TargetRegistry.h>

namespace pacxx {
namespace v2 {
class NativeBackend {
public:
  NativeBackend();

  ~NativeBackend();

  void prepareModule(llvm::Module &M);

  llvm::legacy::PassManager &getPassManager();

  llvm::Module *compile(std::unique_ptr<llvm::Module> &M);

  void *getKernelFptr(llvm::Module *module, const std::string name);

  static std::unique_ptr<llvm::Module> createModule(llvm::LLVMContext &Context,
                                                    const std::string IR);

private:
  llvm::SmallVector<std::string, 10> getTargetFeatures();
  void linkInModule(std::unique_ptr<llvm::Module> &M);
  void applyPasses(llvm::Module &M);

private:
  llvm::legacy::PassManager _PM;
  llvm::TargetMachine *_machine;
  std::unique_ptr<llvm::Module> _composite;
  llvm::ExecutionEngine *_JITEngine;
  bool _pmInitialized;
  bool _disableVectorizer;
  bool _disableSelectEmitter;
};
}
}
#endif // PACXX_V2_NATIVEBACKEND_H
