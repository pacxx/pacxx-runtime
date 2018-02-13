//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <cassert>
#include <iostream>
#include <set>
#include <sstream>
#include <vector>

#include "llvm/IR/Dominators.h"
#include <string>

#include "pacxx/detail/common/transforms/PACXXTransforms.h"
#include "pacxx/detail/common/transforms/CallVisitor.h"
#include "pacxx/detail/common/transforms/ModuleHelper.h"

using namespace llvm;
using namespace std;
using namespace pacxx;

namespace {
class IntrinsicScheduler : public InstVisitor<IntrinsicScheduler> {
public:
  IntrinsicScheduler() {}

  void initialize() { intrinsicClones.clear(); }

  void finalize() {
    for (auto I : intrinsicClones) {
      auto *clone = I.second->clone();
      clone->insertBefore(I.first);
      for (unsigned i = 0; i != I.first->getNumOperands(); ++i) {
        auto op = I.first->getOperand(i);
        if (op == I.second) {
          I.first->setOperand(i, clone);
        }
      }
    }
  }

  void visitCallInst(CallInst &CI) {
    if (auto II = dyn_cast<IntrinsicInst>(&CI)) {
      if (isPACXXIntrinsic(II->getIntrinsicID())) {
        for (auto u : CI.users()) {
          if (Instruction *I = dyn_cast<Instruction>(u)) {
            // clone them if they are not in the same basic block
            if (!isa<PHINode>(I) && I->getParent() != CI.getParent()) {
              intrinsicClones.push_back(make_pair(I, &CI));
            }
          }
        }
      }
    }
  }


private:
  vector<pair<Instruction *, CallInst *>> intrinsicClones;
  vector<CallInst *> dead;
  vector<pair<CallInst *, BinaryOperator *>> repl;
};


struct IntrinsicSchedulerWrapperPass : public ModulePass {
  static char ID;
  IntrinsicSchedulerWrapperPass() : ModulePass(ID) {}
  virtual ~IntrinsicSchedulerWrapperPass() {}

  virtual bool runOnModule(Module &M) {
    IntrinsicScheduler scheduler;

    for (auto &F : M.getFunctionList()) {
      scheduler.initialize();
      scheduler.visit(F);
      scheduler.finalize();
    }

    return true;
  }
};

char IntrinsicSchedulerWrapperPass::ID = 0;
static RegisterPass<IntrinsicSchedulerWrapperPass> X("pacxx-intrinsic-scheduler", "Pass to clone PACXX intrinsics to their direct users", false, false);
}

namespace pacxx {
Pass *createIntrinsicSchedulerPass() { return new IntrinsicSchedulerWrapperPass(); }
}
