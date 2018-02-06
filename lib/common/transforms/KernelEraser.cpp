//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <vector>

#define DEBUG_TYPE "pacxx_kernel_eraser"

#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "pacxx/detail/common/transforms/ModuleHelper.h"
#include "pacxx/detail/common/transforms/PACXXTransforms.h"

using namespace llvm;
using namespace std;
using namespace pacxx;

namespace pacxx {

struct KernelEraserPass : public ModulePass {
  static char ID;
  KernelEraserPass() : ModulePass(ID) {}
  virtual ~KernelEraserPass() {}
  virtual bool runOnModule(Module &M);

private:
  void cleanFromKernels(Module &M);
};

bool KernelEraserPass::runOnModule(Module &M) { 
    cleanFromKernels(M);
    return true; 
}

void KernelEraserPass::cleanFromKernels(Module &M) {
  auto kernels = pacxx::getKernels(&M);

  for (auto F : kernels) {
    F->replaceAllUsesWith(UndefValue::get(F->getType()));
    F->eraseFromParent();
  }
}

char KernelEraserPass::ID = 0;
static RegisterPass<KernelEraserPass>
    X("pacxx-kernel-eraser", "deletes kernels from IR", false, false);
} // namespace pacxx

namespace pacxx {
Pass *createKernelEraserPass() { return new KernelEraserPass(); }
} // namespace pacxx
