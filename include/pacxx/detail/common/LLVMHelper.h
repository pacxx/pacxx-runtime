//
// Created by mhaidl on 30/06/16.
//

#ifndef PACXX_V2_LLVMHELPER_H
#define PACXX_V2_LLVMHELPER_H

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/InstVisitor.h>
#include <llvm/IR/Instructions.h>

namespace pacxx {
namespace common {
class CallFinder : public llvm::InstVisitor<CallFinder> {
public:
  void reset() { found.clear(); }

  void setIntrinsicID(llvm::Intrinsic::ID id) { IID = id; }

  void setOpenCLFunction(llvm::Function *oclF, unsigned v) {
    F = oclF;
    value = v;
  }

  void visitCallInst(llvm::CallInst &I) {
    if (I.isInlineAsm())
      return;
    auto cand = I.getCalledFunction();
    if (cand) {
      if (cand->isIntrinsic()) {
        if (cand->getIntrinsicID() == IID)
          found.push_back(&I);
      } else {
        if (F == cand) {
          auto op = I.getArgOperand(0);
          if (auto CI = dyn_cast<llvm::ConstantInt>(op)) {
            if (static_cast<unsigned>(*(CI->getValue().getRawData())) == value)
              found.push_back(&I);
          }
        }
      }
    }
  }

  const auto &getFoundCalls() { return found; }

private:
  llvm::Intrinsic::ID IID;
  llvm::Function *F;
  unsigned value;
  std::vector<llvm::CallInst *> found;
};
}
}
#endif // PACXX_V2_LLVMHELPER_H
