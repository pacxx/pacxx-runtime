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

static void replaceOCLConfigCall(Function* F, unsigned value) {
      for (auto U : F->users()){
        if (auto CI = dyn_cast<CallInst>(U)){
          CI->replaceAllUsesWith(ConstantInt::get(CI->getType(), value));
        }
      }
}

static Function *cloneKernelForLaunchConfig(Function *F) {

  SmallVector<Type *, 14> Params;
  for (unsigned i = 0; i < 6; ++i)
    Params.push_back(Type::getInt32Ty(F->getContext()));
  for (auto &arg : F->args())
    Params.push_back(arg.getType());

  Type *RetTy = F->getReturnType();
  FunctionType *NFTy = FunctionType::get(RetTy, Params, false);
  auto name = F->getName().str();
  F->setName("undead");
  auto NF = Function::Create(NFTy, F->getLinkage(), name, F->getParent());
  auto DestI = NF->arg_begin() + 6;
  ValueToValueMapTy VMap;
  for (auto I = F->arg_begin(); I != F->arg_end(); ++I) {
    DestI->setName(I->getName());
    VMap[cast<Value>(I)] = cast<Value>(DestI++);
  }
  SmallVector<ReturnInst *, 8> returns;
  CloneFunctionInto(NF, F, VMap, true, returns);
  if (auto MD = F->getParent()->getNamedMetadata("nvvm.annotations")) {
    for (unsigned i = 0; i != MD->getNumOperands(); ++i) {
      auto Op = MD->getOperand(i);
      if (Op->getOperand(0)) {
        if (auto *KF = dyn_cast<Function>(
                dyn_cast<ValueAsMetadata>(Op->getOperand(0).get())
                    ->getValue())) {
          if (KF == F) {
            Op->replaceOperandWith(0, ValueAsMetadata::get(NF));
          }
        }
      }
    }
  }
  F->eraseFromParent();
  return NF;
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

    for (auto &GV : M.getGlobalList()) {
      if (GV.getType()->isPointerTy() && GV.getType()->getAddressSpace() == 3) {
        auto newName = replaceSubstring(GV.getName(), ".", "_");
        GV.setName(newName);
      }
    }

    if (auto F = M.getFunction("__oclc_ISA_version")){
      replaceOCLConfigCall(F, GFX); 
    }

    if (auto F = M.getFunction("__oclc_daz_opt")){
      replaceOCLConfigCall(F, 0); // disable flush to zero
    }

    if (auto F = M.getFunction("__oclc_unsafe_math_opt")){
      replaceOCLConfigCall(F, 0); // disable unsafe math opts
    }
    
    if (auto F = M.getFunction("__oclc_correctly_rounded_sqrt32")){
      replaceOCLConfigCall(F, 0); 
    }

    // create a wrapper for quering the SM base pointer 
    // TODO: outline to a function
    auto VoidTy = Type::getInt8Ty(M.getContext());
    auto SMFTy = FunctionType::get(VoidTy->getPointerTo(3), false);
    Function* SMF = dyn_cast<Function>(M.getOrInsertFunction("__get_extern_shared_mem_ptr", SMFTy));
    assert(SMF && "something went wrong here!");
    auto BB = BasicBlock::Create(M.getContext(), "entry", SMF); 
    IRBuilder<> builder(BB);
    auto Decl =
            Intrinsic::getDeclaration(&M,
                                      Intrinsic::amdgcn_groupstaticsize);
    auto Addr = builder.CreateCall(Decl); 
    auto GEP = builder.CreateGEP(ConstantPointerNull::get(VoidTy->getPointerTo(3)), Addr);
    builder.CreateRet(GEP); 
    
    auto kernels = pacxx::getKernels(&M);

    for (auto &F : kernels)
      cloneKernelForLaunchConfig(F);

    // updated for cloned kernels
    kernels = pacxx::getKernels(&M);

    for (auto &F : kernels) {
      F->setCallingConv(CallingConv::AMDGPU_KERNEL);
      F->setLinkage(GlobalValue::LinkageTypes::ExternalLinkage);
      F->setVisibility(GlobalValue::VisibilityTypes::DefaultVisibility);
    }

    for (auto &F : M){ 
      if (!F.isIntrinsic())
        F.setAttributes(AttributeList());
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
