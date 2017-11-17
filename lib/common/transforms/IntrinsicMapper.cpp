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

#define DEBUG_TYPE "pacxx_emit_select"

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "pacxx/detail/common/transforms/ModuleHelper.h"
#include "pacxx/detail/common/transforms/PACXXTransforms.h"

using namespace llvm;
using namespace std;
using namespace pacxx;

namespace llvm {
void initializeIntrinsicMapperPass(PassRegistry &);
}

namespace {

static bool isPACXXIntrinsic(Intrinsic::ID id) {
  switch (id) {
  case Intrinsic::pacxx_barrier0:
  case Intrinsic::pacxx_read_ntid_x:
  case Intrinsic::pacxx_read_ntid_y:
  case Intrinsic::pacxx_read_ntid_z:
  case Intrinsic::pacxx_read_ntid_w:
  case Intrinsic::pacxx_read_tid_x:
  case Intrinsic::pacxx_read_tid_y:
  case Intrinsic::pacxx_read_tid_z:
  case Intrinsic::pacxx_read_tid_w:
  case Intrinsic::pacxx_read_ctaid_x:
  case Intrinsic::pacxx_read_ctaid_y:
  case Intrinsic::pacxx_read_ctaid_z:
  case Intrinsic::pacxx_read_ctaid_w:
  case Intrinsic::pacxx_read_nctaid_x:
  case Intrinsic::pacxx_read_nctaid_y:
  case Intrinsic::pacxx_read_nctaid_z:
  case Intrinsic::pacxx_read_nctaid_w:
  case Intrinsic::pacxx_backend_id:
    return true;
  default:
    break;
  }
  return false;
}

static Value *mapPACXXIntrinsicNative(Module *M, IntrinsicInst *II) {
  switch (II->getIntrinsicID()) {
  case Intrinsic::pacxx_backend_id:
    return ConstantInt::get(II->getType(), 1);
  default:
    break;
  }
  return II;
}

static Value *mapPACXXIntrinsicNVPTX(Module *M, IntrinsicInst *II) {
  Function *mapping = nullptr;

  switch (II->getIntrinsicID()) {
  case Intrinsic::pacxx_backend_id:
    return ConstantInt::get(II->getType(), 0);
  case Intrinsic::pacxx_barrier0:
    mapping = Intrinsic::getDeclaration(M, Intrinsic::nvvm_barrier0);
    break;
  case Intrinsic::pacxx_read_ntid_x:
    mapping =
        Intrinsic::getDeclaration(M, Intrinsic::nvvm_read_ptx_sreg_ntid_x);
    break;
  case Intrinsic::pacxx_read_ntid_y:
    mapping =
        Intrinsic::getDeclaration(M, Intrinsic::nvvm_read_ptx_sreg_ntid_y);
    break;
  case Intrinsic::pacxx_read_ntid_z:
    mapping =
        Intrinsic::getDeclaration(M, Intrinsic::nvvm_read_ptx_sreg_ntid_z);
    break;
  case Intrinsic::pacxx_read_ntid_w:
    mapping =
        Intrinsic::getDeclaration(M, Intrinsic::nvvm_read_ptx_sreg_ntid_w);
    break;
  case Intrinsic::pacxx_read_tid_x:
    mapping = Intrinsic::getDeclaration(M, Intrinsic::nvvm_read_ptx_sreg_tid_x);
    break;
  case Intrinsic::pacxx_read_tid_y:
    mapping = Intrinsic::getDeclaration(M, Intrinsic::nvvm_read_ptx_sreg_tid_y);
    break;
  case Intrinsic::pacxx_read_tid_z:
    mapping = Intrinsic::getDeclaration(M, Intrinsic::nvvm_read_ptx_sreg_tid_z);
    break;
  case Intrinsic::pacxx_read_tid_w:
    mapping = Intrinsic::getDeclaration(M, Intrinsic::nvvm_read_ptx_sreg_tid_w);
    break;
  case Intrinsic::pacxx_read_ctaid_x:
    mapping =
        Intrinsic::getDeclaration(M, Intrinsic::nvvm_read_ptx_sreg_ctaid_x);
    break;
  case Intrinsic::pacxx_read_ctaid_y:
    mapping =
        Intrinsic::getDeclaration(M, Intrinsic::nvvm_read_ptx_sreg_ctaid_y);
    break;
  case Intrinsic::pacxx_read_ctaid_z:
    mapping =
        Intrinsic::getDeclaration(M, Intrinsic::nvvm_read_ptx_sreg_ctaid_z);
    break;
  case Intrinsic::pacxx_read_ctaid_w:
    mapping =
        Intrinsic::getDeclaration(M, Intrinsic::nvvm_read_ptx_sreg_ctaid_w);
    break;
  case Intrinsic::pacxx_read_nctaid_x:
    mapping =
        Intrinsic::getDeclaration(M, Intrinsic::nvvm_read_ptx_sreg_nctaid_x);
    break;
  case Intrinsic::pacxx_read_nctaid_y:
    mapping =
        Intrinsic::getDeclaration(M, Intrinsic::nvvm_read_ptx_sreg_nctaid_y);
    break;
  case Intrinsic::pacxx_read_nctaid_z:
    mapping =
        Intrinsic::getDeclaration(M, Intrinsic::nvvm_read_ptx_sreg_nctaid_z);
    break;
  case Intrinsic::pacxx_read_nctaid_w:
    mapping =
        Intrinsic::getDeclaration(M, Intrinsic::nvvm_read_ptx_sreg_nctaid_w);
    break;
  default:
    break;
  }

  assert(mapping && "No mapping for this intrinsic!");
  II->setCalledFunction(mapping);
  return II;
}

static void mapPACXXIntrinsicToLaunchParamAMDGCN(Module *M, IntrinsicInst *II) {
  auto F = II->getParent()->getParent();
  switch (II->getIntrinsicID()) {
  case Intrinsic::pacxx_read_nctaid_x:
    II->replaceAllUsesWith(F->arg_begin());
    break;
  case Intrinsic::pacxx_read_nctaid_y:
    II->replaceAllUsesWith((F->arg_begin() + 1));
    break;
  case Intrinsic::pacxx_read_nctaid_z:
    II->replaceAllUsesWith((F->arg_begin() + 2));
    break;
  case Intrinsic::pacxx_read_ntid_x:
    II->replaceAllUsesWith((F->arg_begin() + 3));
    break;
  case Intrinsic::pacxx_read_ntid_y:
    II->replaceAllUsesWith((F->arg_begin() + 4));
    break;
  case Intrinsic::pacxx_read_ntid_z:
    II->replaceAllUsesWith((F->arg_begin() + 5));
    break;
  default:
    break;
  }
}

static Value *mapPACXXIntrinsicToSpecialFunctionAMDGCN(Module *M,
                                                       IntrinsicInst *II) {
  llvm_unreachable("not implemented");
  return nullptr;
}

static Value *mapPACXXIntrinsicAMDGCN(Module *M, IntrinsicInst *II) {
  Value *mapping = nullptr;
  switch (II->getIntrinsicID()) {
  case Intrinsic::pacxx_backend_id:
    return ConstantInt::get(II->getType(), 2);
  case Intrinsic::pacxx_barrier0:
    mapping = Intrinsic::getDeclaration(M, Intrinsic::amdgcn_s_barrier);
    break;
  case Intrinsic::pacxx_read_ctaid_x:
    mapping = Intrinsic::getDeclaration(M, Intrinsic::amdgcn_workgroup_id_x);
    break;
  case Intrinsic::pacxx_read_ctaid_y:
    mapping = Intrinsic::getDeclaration(M, Intrinsic::amdgcn_workgroup_id_y);
    break;
  case Intrinsic::pacxx_read_ctaid_z:
    mapping = Intrinsic::getDeclaration(M, Intrinsic::amdgcn_workgroup_id_z);
    break;
  case Intrinsic::pacxx_read_tid_x:
    mapping = Intrinsic::getDeclaration(M, Intrinsic::amdgcn_workitem_id_x);
    break;
  case Intrinsic::pacxx_read_tid_y:
    mapping = Intrinsic::getDeclaration(M, Intrinsic::amdgcn_workitem_id_y);
    break;
  case Intrinsic::pacxx_read_tid_z:
    mapping = Intrinsic::getDeclaration(M, Intrinsic::amdgcn_workitem_id_z);
    break;
  case Intrinsic::pacxx_read_nctaid_x:
  case Intrinsic::pacxx_read_nctaid_y:
  case Intrinsic::pacxx_read_nctaid_z:
  case Intrinsic::pacxx_read_ntid_x:
  case Intrinsic::pacxx_read_ntid_y:
  case Intrinsic::pacxx_read_ntid_z:
    mapPACXXIntrinsicToLaunchParamAMDGCN(M, II);
    return nullptr;
  default:
    return mapPACXXIntrinsicToSpecialFunctionAMDGCN(M, II);
  }

  assert(mapping && "No mapping for this intrinsic!");

  II->setCalledFunction(mapping);

  return II;
}

struct IntrinsicMapper : public ModulePass {
  static char ID;
  IntrinsicMapper() : ModulePass(ID) {
    initializeIntrinsicMapperPass(*PassRegistry::getPassRegistry());
  }
  virtual ~IntrinsicMapper() {}
  virtual bool runOnModule(Module &M) override;
  virtual void getAnalysisUsage(AnalysisUsage &AU) const override;
};

bool IntrinsicMapper::runOnModule(Module &M) {
  bool modified = true;

  struct IntrinsicVisitor : public InstVisitor<IntrinsicVisitor> {

    void visitCallInst(CallInst &CI) {

      if (auto II = dyn_cast<IntrinsicInst>(&CI)) {
        if (isPACXXIntrinsic(II->getIntrinsicID())) {
          if (M->getTargetTriple().find("nvptx") != std::string::npos) {
            if (auto mapped = mapPACXXIntrinsicNVPTX(M, II))
              if (mapped != II)
                replacements.push_back(make_pair(II, mapped));
          } else if (M->getTargetTriple().find("amdgcn") != std::string::npos) {
            if (auto mapped = mapPACXXIntrinsicAMDGCN(M, II))
              if (mapped != II)
                replacements.push_back(make_pair(II, mapped));
          } else {
            if (auto mapped = mapPACXXIntrinsicNative(M, II))
              if (mapped != II)
                replacements.push_back(make_pair(II, mapped));
          }
        }
      }
    }

    Module *M;
    TargetTransformInfo *TTI;
    SmallVector<std::pair<IntrinsicInst *, Value *>, 8> replacements;
  } visitor;

  auto kernels = pacxx::getKernels(&M);
  visitor.M = &M;
  for (auto &F : M) {
    visitor.TTI = &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    visitor.visit(F);
  }
  for (auto P : visitor.replacements) {
    if (P.second != nullptr) {
      if (auto I = dyn_cast<Instruction>(P.second))
        I->insertBefore(P.first);
      P.first->replaceAllUsesWith(P.second);
    }
    P.first->eraseFromParent();
  }

  return modified;
}

void IntrinsicMapper::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetLibraryInfoWrapperPass>();
  AU.addRequired<TargetTransformInfoWrapperPass>();
}

} // namespace

char IntrinsicMapper::ID = 0;

INITIALIZE_PASS_BEGIN(
    IntrinsicMapper, "pacxx-intrin-mapper",
    "PACXXSelectEmitter: transform masked intrinsics to selects", true, true)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(IntrinsicMapper, "pacxx-emit-select",
                    "IntrinsicMapper: transform pacxx intrinsics to target "
                    "dependend intrinsics",
                    true, true)

namespace pacxx {
Pass *createIntrinsicMapperPass() { return new IntrinsicMapper(); }
} // namespace pacxx
