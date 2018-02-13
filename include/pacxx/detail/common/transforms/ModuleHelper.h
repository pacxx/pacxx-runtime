//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORM_PACXX_MODULEHELPER_H
#define LLVM_TRANSFORM_PACXX_MODULEHELPER_H

#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "PACXXTransforms.h"
#include "llvm/Support/raw_ostream.h"
#include "CallVisitor.h"

#include <set>
#include <string>

#define STD_VECTOR_TYPE "class.std::__1::vector"

using namespace llvm;

namespace pacxx {


template <typename T> void ReplaceUnsafe(T *from, T *to) {
  if (from == to)
    return;
  while (!from->use_empty()) {
    auto &U = *from->use_begin();
    U.set(to);
  }
  from->eraseFromParent();
}

bool isPACXXIntrinsic(Intrinsic::ID id);

void cleanupDeadCode(Module *M);

bool isOpenCLFunction(StringRef name);

void fixVectorOffset(Function *F, unsigned offset);

struct CEExtractor : public InstVisitor<CEExtractor> {
  void visitInstruction(Instruction &I) {
    if(!isa<PHINode>(I)){
      for (unsigned i = 0; i < I.getNumOperands(); ++i) {
        auto op = I.getOperand(i);
        if (auto CE = dyn_cast<ConstantExpr>(op)) {
          auto opi = CE->getAsInstruction();
          opi->insertBefore(&I);
          I.setOperand(i, opi);
          visitInstruction(*opi);
        }
      }
    }
  }
};

}

#endif
