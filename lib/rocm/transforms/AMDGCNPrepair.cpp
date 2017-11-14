/* Copyright (C) University of Muenster - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Michael Haidl <michael.haidl@uni-muenster.de>, 2010-2014
 */

#include <cassert>
#include <iostream>
#include <vector>

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

static void createISAVersionFunction(Module *M, unsigned GFX) {
  // create an ISA version function
  auto FTy =
      FunctionType::get(cast<Type>(Type::getInt32Ty(M->getContext())), false);
  auto ISAFunc =
      Function::Create(FTy, GlobalValue::LinkageTypes::LinkOnceODRLinkage,
                       "__oclc_ISA_version", M);
  ISAFunc->setVisibility(GlobalValue::VisibilityTypes::ProtectedVisibility);

  IRBuilder<> builder(&ISAFunc->front());
  builder.CreateRet(builder.getInt32(GFX));
}

struct AMDGCNPrepair : public ModulePass {
  static char ID;
  unsigned GFX;
  AMDGCNPrepair(unsigned GFX = 803) : ModulePass(ID), GFX(GFX) {}
  virtual ~AMDGCNPrepair() {}

  virtual bool runOnModule(Module &M) {
    bool modified = true;

    M.setTargetTriple("amdgcn--amdhsa-amdgiz");

    M.setDataLayout("e-p:64:64-p1:64:64-p2:64:64-p3:32:32-p4:32:32-p5:32:32-"
                    "i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:"
                    "256-v512:512-v1024:1024-v2048:2048-n32:64-A5");

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

    createISAVersionFunction(&M, GFX);

    auto kernels = pacxx::getKernels(&M);

    for (auto &F : kernels) {
      F->setCallingConv(CallingConv::AMDGPU_KERNEL);
      F->setLinkage(GlobalValue::LinkageTypes::ExternalLinkage);
      F->setVisibility(GlobalValue::VisibilityTypes::DefaultVisibility);
      F->setAttributes(AttributeList());
    }

    return modified;
  }
};

char AMDGCNPrepair::ID = 0;
static RegisterPass<AMDGCNPrepair>
    X("pacxx-amdgcn-prepair", "Prepairs module for AMDGCN ISA generation",
      false, false);
} // namespace

namespace pacxx {
Pass *createAMDGCNPrepairPass(unsigned GFX = 803) {
  return new AMDGCNPrepair(GFX);
}
} // namespace pacxx
