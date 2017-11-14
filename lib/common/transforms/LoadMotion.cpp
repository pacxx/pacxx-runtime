//===-----------------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "pacxx-load-motion"

#include "llvm/ADT/SetVector.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

#include "pacxx/detail/common/transforms/ModuleHelper.h"

using namespace llvm;
using namespace std;
using namespace pacxx;

namespace {

static bool
moveLoads(ArrayRef<std::pair<GetElementPtrInst *, GetElementPtrInst *>> geps) {
  for (auto P : geps) {
    auto I = P.first;
    auto J = P.second;
    Value* load = nullptr; 
    Value *candidate;
    BasicBlock *BB = nullptr;
    // find dominating   
    if (std::find(pred_begin(J->getParent()), pred_end(J->getParent()),
                  I->getParent()) != pred_end(J->getParent())) {
      BB = I->getParent();
      candidate = J;
      for (auto U : I->users()){
        if (isa<LoadInst>(U)){
          load = U; 
          break;
        }
      }
    } else if (std::find(pred_begin(I->getParent()), pred_end(I->getParent()),
                         J->getParent()) != pred_end(I->getParent())) {
      BB = J->getParent();
      candidate = I;
      for (auto U : J->users()){
        if (isa<LoadInst>(U)){
          load = U; 
          break;
        }
      }
    } else if (I->getParent() == J->getParent()) {
      BB = I->getParent();
      candidate = J;
      for (auto U : I->users()){
        if (isa<LoadInst>(U)){
          load = U; 
          break;
        }
      }
    }

    if (!BB) // opt out
      continue;
    candidate->dump();
    I->dump();
    J->dump();
    if (candidate->hasNUses(1)){
      candidate = *candidate->user_begin();
      if (isa<BitCastInst>(candidate)){
        Value* user = *candidate->user_begin();
        if (auto LI = dyn_cast<LoadInst>(user)){
          load->dump();
          LI->dump();
          I->dump();
          J->dump();
        }
      }
    }
  }
  return false;
}

static SmallVector<std::pair<GetElementPtrInst *, GetElementPtrInst *>, 5>
collectGEPPairs(const SetVector<GetElementPtrInst *> &geps) {
  SmallVector<std::pair<GetElementPtrInst *, GetElementPtrInst *>, 5> pairs;

  for (auto I : geps) {
    for (auto J : geps) {
      if (I == J)
        continue;
      bool match = I->getNumIndices() == J->getNumIndices();
    
      if (match) { // check if number of indices match
        match = std::equal(I->idx_begin(), I->idx_end(), J->idx_begin());
        if (match) // now check if the pointer operands match
          match = I->getPointerOperand() == J->getPointerOperand();
        if (match) // finally check if the types match in width
          match =
              I->getType()->getPointerElementType()->getScalarSizeInBits() ==
              J->getType()->getPointerElementType()->getScalarSizeInBits();
      }    
      if (match)
        if (std::find_if(pairs.begin(), pairs.end(), [&](auto &p) {
              return (p.first == I && p.second == J) ||
                     (p.first == J && p.second == I);
            }) == pairs.end()){
            I->dump(); 
            J->dump();
          pairs.push_back(
              std::pair<GetElementPtrInst *, GetElementPtrInst *>(I, J));
              llvm::errs() << "\n";
        }
    }
  }
  return pairs;
}

struct LoadMotion : public ModulePass {
  static char ID;
  LoadMotion() : ModulePass(ID) {}
  virtual ~LoadMotion() {}

  virtual bool runOnModule(Module &M) {

    struct GEPCollector : public InstVisitor<GEPCollector> {
      SetVector<GetElementPtrInst *> geps;

      void visitGetElementPtrInst(GetElementPtrInst &I) { geps.insert(&I); }

    } collector;

    auto kernels = getKernels(&M);

    for (auto F : kernels) {
      collector.geps.clear();
      collector.visit(F);
      auto geps = collectGEPPairs(collector.geps);
      llvm::errs() << "Mateched " << geps.size() << "\n";
      moveLoads(geps);
    }

    return false;
  }
};

char LoadMotion::ID = 0;
static RegisterPass<LoadMotion> X("pacxx-load-motion",
                                  "Moves load instructions to predecessor "
                                  "blocks and removes redundant loads.",
                                  false, false);
}

namespace pacxx {
Pass *createLoadMotionPass() { return new LoadMotion(); }
}
