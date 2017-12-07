//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef PACXX_V2_MSPENGINE_H
#define PACXX_V2_MSPENGINE_H

#include "pacxx/detail/Kernel.h"
#include "pacxx/detail/common/Log.h"
#include "pacxx/detail/common/Meta.h"
#include <set>

#include "../KernelArgument.h"

namespace llvm {
	class Module; 
	class Function;
	class ExecutionEngine;
}

namespace pacxx {
namespace v2 {
class MSPEngine {
public:
  using stub_ptr_t = void (*)(void *, void *);
  using i64_stub_ptr_t = size_t (*)(size_t);

  MSPEngine();

  void initialize(std::unique_ptr<llvm::Module> M);
  void evaluate(const llvm::Function &KF, Kernel &kernel);

  size_t getArgBufferSize(const llvm::Function &KF, Kernel &kernel);
  void transformModule(llvm::Module &M, Kernel &K);
  bool isDisabled();

private:
  bool _disabled;
  llvm::ExecutionEngine *_engine;
  llvm::Module *_mspModule;
  std::set<std::pair<llvm::Function *, int>> _stubs;
};
}
}
#endif // PACXX_V2_MSPENGINE_H
