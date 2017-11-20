//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "pacxx/detail/common/transforms/ModuleHelper.h"

#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;
using namespace std;
using namespace pacxx;

namespace {

enum class PFStyle { nvidia, native, amd };

static void handlePrintfCall(CallInst *I, PFStyle flavour) {
  auto F = I->getCalledFunction();

  assert(F && "CallInst does not have a valid Function");

  switch (flavour) {
  case PFStyle::nvidia:
    if (auto GEP = dyn_cast<ConstantExpr>(I->getOperand(0))) {
      if (auto str = dyn_cast<GlobalVariable>(GEP->getOperand(0))) {
        str->mutateType(
            str->getType()->getPointerElementType()->getPointerTo(4));
        if (!str->getMetadata("pacxx.as.constant"))
          str->setMetadata("pacxx.as.constant",
                           llvm::MDNode::get(F->getContext(), nullptr));
        auto c0 =
            ConstantInt::get(Type::getInt64Ty(F->getParent()->getContext()), 0);
        vector<Value *> idx;
        idx.push_back(c0);
        idx.push_back(c0);
        auto newGEP = GetElementPtrInst::Create(
            str->getType()->getElementType(), str, idx, "", I);
        auto ASC = AddrSpaceCastInst::CreatePointerBitCastOrAddrSpaceCast(
            newGEP, newGEP->getType()->getPointerElementType()->getPointerTo(),
            "", I);
        I->setOperand(0, ASC);
      }
    }
    break;
  case PFStyle::native:
    if (auto GEP = dyn_cast<ConstantExpr>(I->getOperand(0))) {
      if (auto str = dyn_cast<GlobalVariable>(GEP->getOperand(0))) {
        if (!str->getMetadata("pacxx.as.constant"))
          str->setMetadata("pacxx.as.constant",
                           llvm::MDNode::get(F->getContext(), nullptr));
        vector<Value *> idx;
        auto c0 =
            ConstantInt::get(Type::getInt64Ty(F->getParent()->getContext()), 0);
        idx.push_back(c0);
        idx.push_back(c0);
        auto newGEP = GetElementPtrInst::Create(
            str->getType()->getElementType(), str, idx, "", I);
        I->setOperand(0, newGEP);
      }
    }
    break;
  case PFStyle::amd:
    llvm_unreachable("not implemented");
  }
}

struct PACXXCodeGenPrepare : public ModulePass {
  static char ID;
  PACXXCodeGenPrepare() : ModulePass(ID) {}
  virtual ~PACXXCodeGenPrepare() {}

  virtual bool runOnModule(Module &M) {

    auto visitor = make_CallVisitor([&](CallInst *I) {
      if (!I && I->isInlineAsm())
        return;

      if (!isa<Function>(I->getCalledValue())) {
        return;
      }

      auto F = I->getCalledFunction();

      if (!F)
        return;

      // mark all called functions as always inline to pull them into the
      // kernel
      if (F->hasFnAttribute(llvm::Attribute::NoInline))
        F->removeFnAttr(llvm::Attribute::NoInline);
      if (!F->hasFnAttribute(llvm::Attribute::AlwaysInline))
        F->addFnAttr(llvm::Attribute::AlwaysInline);
      if (F->hasFnAttribute(llvm::Attribute::OptimizeNone))
        F->removeFnAttr(llvm::Attribute::OptimizeNone);

      if (I->getCalledFunction()->getName().find("pacxx6nvidia6printf") !=
          StringRef::npos) {
        handlePrintfCall(I, PFStyle::nvidia);
      } else if (I->getCalledFunction()->getName().find("printf") !=
                     StringRef::npos ||
                 I->getCalledFunction()->getName().find("puts") !=
                     StringRef::npos) {
        handlePrintfCall(I, PFStyle::native);
      }
    });

    auto kernels = pacxx::getKernels(&M);
    for (auto &F : kernels)
      visitor.visit(F);

    cleanupDeadCode(&M);

    return true;
  }
}; // namespace

char PACXXCodeGenPrepare::ID = 0;
static RegisterPass<PACXXCodeGenPrepare>
    X("pacxx-codegen-prepare", "Prepares a Module for PACXX Code Generation",
      false, false);
} // namespace

namespace pacxx {
Pass *createPACXXCodeGenPrepare() { return new PACXXCodeGenPrepare(); }
} // namespace pacxx
