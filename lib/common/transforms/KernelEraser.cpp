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
  void cleanFromPACXXIntrinsics(Module &M);
};

bool KernelEraserPass::runOnModule(Module &M) {
  cleanFromKernels(M);
  cleanFromPACXXIntrinsics(M);

  return true;
}

void KernelEraserPass::cleanFromPACXXIntrinsics(Module &M) {
  SmallVector<Function *, 8> dead;
  for (auto &F : M) {
    if (F.isIntrinsic()) {
      if (isPACXXIntrinsic(F.getIntrinsicID())) {
        dead.push_back(&F);
      }
    }
  }

  for (auto F : dead) {
    SmallVector<Instruction *, 8> dead_users;
    for (auto U : F->users()) {
      if (isa<IntegerType>(U->getType())) {
        U->replaceAllUsesWith(ConstantInt::get(U->getType(), 0, false));
      } 
      else
        U->dump();
      dead_users.push_back(cast<Instruction>(U));
    }
    for (auto U : dead_users)
      U->eraseFromParent();
    // F->replaceAllUsesWith(UndefValue::get(F->getType()));
    F->eraseFromParent();
  }
}

void KernelEraserPass::cleanFromKernels(Module &M) {
  auto kernels = pacxx::getKernels(&M);

  struct CallInstVisitor : public InstVisitor<CallInstVisitor> {
    SmallVector<Instruction *, 8> dead;
    void visitCallInst(CallInst &CI) { dead.push_back(&CI); }
  } calls;

  for (auto F : kernels) {
    calls.visit(F);
  }
  for (auto CI : calls.dead) {
    CI->replaceAllUsesWith(UndefValue::get(CI->getType()));
    CI->eraseFromParent();
  }
}

char KernelEraserPass::ID = 0;
static RegisterPass<KernelEraserPass>
    X("pacxx-kernel-eraser", "deletes kernels from IR", false, false);
} // namespace pacxx

namespace pacxx {
Pass *createKernelEraserPass() { return new KernelEraserPass(); }
} // namespace pacxx
