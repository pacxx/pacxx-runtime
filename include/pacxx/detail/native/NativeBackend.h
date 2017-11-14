//===-----------------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

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

  std::unique_ptr<llvm::Module> prepareModule(llvm::Module &M);

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
  bool _disableExpPasses;
};
}
}
#endif // PACXX_V2_NATIVEBACKEND_H
