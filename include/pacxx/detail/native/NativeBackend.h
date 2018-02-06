//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef PACXX_V2_NATIVEBACKEND_H
#define PACXX_V2_NATIVEBACKEND_H
#include <memory>
#include <string>

namespace llvm {
	class Module; 
	class LLVMContext;
	class ExecutionEngine;
	class TargetMachine;
}

namespace pacxx {
namespace v2 {
class NativeBackend {
public:
  NativeBackend();

  ~NativeBackend();

  std::unique_ptr<llvm::Module> prepareModule(llvm::Module &M);

  llvm::Module *compile(std::unique_ptr<llvm::Module> &M);

  void *getKernelFptr(llvm::Module *module, const std::string name);

  static std::unique_ptr<llvm::Module> createModule(llvm::LLVMContext &Context,
                                                    const std::string IR);

private:
  void linkInModule(std::unique_ptr<llvm::Module> &M);
  void applyPasses(llvm::Module &M);

private:
  llvm::TargetMachine *_machine;
  std::unique_ptr<llvm::Module> _composite;
  llvm::ExecutionEngine *_JITEngine;
  bool _disableVectorizer;
  bool _disableSelectEmitter;
  bool _disableExpPasses;
};
}
}
#endif // PACXX_V2_NATIVEBACKEND_H
