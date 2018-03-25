//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <iostream>
#include <vector>

#define DEBUG_TYPE "nvvm-transformation"

#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
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

#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include "pacxx/detail/common/transforms/CallVisitor.h"
#include "pacxx/detail/common/transforms/ModuleHelper.h"

using namespace llvm;
using namespace std;
using namespace pacxx;

namespace {

struct NVPTXPrepair : public ModulePass {
  static char ID;
  NVPTXPrepair() : ModulePass(ID) {}
  virtual ~NVPTXPrepair() {}

  virtual bool runOnModule(Module &M) {
    bool modified = true;

    unsigned ptrSize = M.getDataLayout().getPointerSizeInBits();

    if (ptrSize == 64){
      M.setTargetTriple("nvptx64-unknown-unknown");

      M.setDataLayout("e-i64:64-i128:128-v16:16-v32:32-n16:32:64");
    }
    else {
      M.setTargetTriple("nvptx-unknown-unknown");

      M.setDataLayout("e-i32:32-i128:128-v16:16-v32:32-n16:32:64");
    }

    auto replaceSubstring = [](string Str, const StringRef &From,
                               const StringRef &To) {
      size_t Pos = 0;
      while ((Pos = Str.find(From, Pos)) != std::string::npos) {
        Str.replace(Pos, From.size(), To.data(), To.size());
        Pos += To.size();
      }
      return Str;
    };

    // replace the . to _ according to the PTX standard
    for (auto &GV : M.getGlobalList()) {
      if (GV.getType()->isPointerTy() && GV.getType()->getAddressSpace() == 3) {
        auto newName = replaceSubstring(GV.getName(), ".", "_");
        GV.setName(newName);
      }
    }

    auto kernels = pacxx::getKernels(&M);

    for (auto &F : kernels) {
      if(F->getReturnType()->isVoidTy())
      	F->setCallingConv(CallingConv::PTX_Kernel);
      else 
	F->setCallingConv(CallingConv::PTX_Device); 
      F->setLinkage(GlobalValue::LinkageTypes::ExternalLinkage);
      F->setVisibility(GlobalValue::VisibilityTypes::DefaultVisibility);
    }

    return modified;
  }
};

char NVPTXPrepair::ID = 0;
static RegisterPass<NVPTXPrepair> X("pacxx-nvptx-prepare", "Prepairs module for PTX generation", false, false);
}

namespace pacxx {
Pass *createNVPTXPrepairPass() { return new NVPTXPrepair(); }
}
