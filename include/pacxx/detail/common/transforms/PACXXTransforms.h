//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include <llvm/IR/InstVisitor.h>
#include <llvm/IR/Module.h>
#include <map>
#include <set>
#include <vector>

#include <memory>
using namespace llvm;
using namespace std;

namespace pacxx {

struct MSPGeneration : public ModulePass {
  static char ID;
  MSPGeneration();
  virtual ~MSPGeneration();
  virtual bool runOnModule(Module &M);
  std::unique_ptr<Module> getReflectionModule();

private:
  bool runOnModuleAtCompileTime(Module &M);

  bool runOnModuleAtRunTime(Module &M);

  void cleanFromKerneles(Module &M);

  void cleanFromReflections(Module &M);

  class ReflectionHandler : public InstVisitor<ReflectionHandler> {
  public:
    ReflectionHandler(Module *module);

    void visitCallInst(CallInst &CI);

    Function *createCallStub(CallInst &CI, int c);
    Function *createCallWrapper(Function *F, int c);

    void finalize();

  private:
    Module *M;
    set<Function *> reflects;
    vector<pair<CallInst *, int>> stubs;
    map<CallInst *, CallInst *> replacements;
    int count;
  };

  class DeadInstructionHandler : public InstVisitor<DeadInstructionHandler> {
  public:
    void visitInstruction(Instruction &I) {
      if (I.isTerminator())
        return;

      if (isa<StoreInst>(&I))
        return;

      if (I.hasNUses(0))
        dead.push_back(&I);
    }

    void finalize() {
      Function *F = nullptr;
      if (!dead.empty())
        F = dead.front()->getParent()->getParent();

      for (auto I : dead)
        I->eraseFromParent();

      do {
        dead.clear();
        visit(F);
        for (auto I : dead)
          I->eraseFromParent();
      } while (!dead.empty());
    }

  private:
    vector<Instruction *> dead;
  };

  class ReturnVisitor : public InstVisitor<ReturnVisitor> {
  public:
    void visitReturnInst(ReturnInst &I) {
      auto &Ctx = I.getContext();
      if (!I.getReturnValue()) {
        // I.dump();
        auto C = ConstantInt::get(Type::getInt64Ty(Ctx), APInt(64, 0xBAADC0DE));
        IMap[&I] = ReturnInst::Create(Ctx, C, &I);
      }
    }

    void visitBranchInst(BranchInst &I) {
      // if (&I != exit)
      //{
      // I.dump();
      // IMap[&I] = new UnreachableInst(I.getContext(), &I);
      //}
    }

    void runOn(Function &F, BranchInst *exit) {
      visit(&F);
      this->exit = exit;
      for (auto p : IMap) {
        p.first->eraseFromParent();
      }

      IMap.clear();
    }

  private:
    BranchInst *exit;
    map<Instruction *, Instruction *> IMap;
  };

  std::unique_ptr<Module> RM;
};

template <typename PTy = std::pair<Function *, int>>
PTy getTagedFunction(MDNode *MD, StringRef desc) {
  if (MD->getOperand(0) != nullptr) {
    auto *F = dyn_cast<Function>(
        dyn_cast<ValueAsMetadata>(MD->getOperand(0).get())->getValue());
    if (desc == "")
      return PTy(F, 0);
    else {
      if (auto *str = dyn_cast<MDString>(MD->getOperand(1))){
        if (str->getString() == desc) {
          if (auto *Val = dyn_cast<ConstantInt>(
                  dyn_cast<ValueAsMetadata>(MD->getOperand(2).get())
                      ->getValue()))
            return PTy(F, *(Val->getValue().getRawData()));

          assert(false && "third operand on MDNode is not i32Ty");
        }
      }
    }
  }
  return PTy(nullptr, -1);
}

template <typename CTy = std::set<Function *>>
CTy getTagedFunctions(Module *M, StringRef twine, StringRef desc) {
  CTy functions;
  if (auto MD = M->getNamedMetadata(twine)) {
    for (unsigned i = 0; i != MD->getNumOperands(); ++i) {
      auto p = getTagedFunction(MD->getOperand(i), desc);
      if (p.first)
        functions.insert(p.first);
    }
  }
  return functions;
}

template <typename CTy = std::set<std::pair<Function *, int>>>
CTy getTagedFunctionsWithTag(Module *M, StringRef twine, StringRef desc) {
  CTy functions;
  if (auto MD = M->getNamedMetadata(twine)) {
    for (unsigned i = 0; i != MD->getNumOperands(); ++i) {
      auto p = getTagedFunction(MD->getOperand(i), desc);
      if (p.first)
        functions.insert(p);
    }
  }
  return functions;
}

template <typename CTy = std::set<Function *>> CTy getKernels(Module *M) {
  auto functions = getTagedFunctions(M, "pacxx.kernel", "");
  return functions;
}

} // namespace pacxx
