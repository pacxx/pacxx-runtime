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
  void cleanFromSharedMemory(Module &M);
};

bool KernelEraserPass::runOnModule(Module &M) {
  cleanFromKernels(M);
  cleanFromSharedMemory(M);
  cleanFromPACXXIntrinsics(M);

  return true;
}

void KernelEraserPass::cleanFromSharedMemory(Module &M){
  SmallVector<GlobalValue*, 8> sm;
  for (auto &GV : M.global_values()){
    if (auto GVar = dyn_cast<GlobalVariable>(&GV)){
      if (GVar->hasMetadata() && GVar->getMetadata("pacxx.as.shared") != nullptr){
        GV.replaceAllUsesWith(UndefValue::get(GV.getType()));
        sm.push_back(&GV);
      }
    }
  }
  for (auto& GV : sm)
    GV->eraseFromParent();
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
    SmallVector<CallInst *, 1> calls;
    SmallVector<InvokeInst *, 1> invokes;
    void visitCallInst(CallInst &CI) { calls.push_back(&CI); }
    void visitInvokeInst(InvokeInst &II) { invokes.push_back(&II); }
  } dead;

  for (auto F : kernels) {
    dead.visit(F);
  }
  for (auto CI : dead.calls) {
    CI->replaceAllUsesWith(UndefValue::get(CI->getType()));
    CI->eraseFromParent();
  }

  for (auto II : dead.invokes) {
    // branch if no execption was catched
    BranchInst::Create(II->getNormalDest(), II);

    II->replaceAllUsesWith(UndefValue::get(II->getType()));
    auto LP = II->getLandingPadInst();
    LP->replaceAllUsesWith(UndefValue::get(LP->getType()));
    LP->eraseFromParent();
    II->getParent()->getParent()->setPersonalityFn(nullptr);
    II->eraseFromParent();
  }

  for (auto F : kernels) {
    F->dump();
  }
}

char KernelEraserPass::ID = 0;
static RegisterPass<KernelEraserPass>
    X("pacxx-kernel-eraser", "deletes kernels from IR", false, false);
} // namespace pacxx

namespace pacxx {
Pass *createKernelEraserPass() { return new KernelEraserPass(); }
} // namespace pacxx
