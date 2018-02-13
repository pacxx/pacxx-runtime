#include "pacxx/detail/common/transforms/ModuleHelper.h"

namespace pacxx{
bool isPACXXIntrinsic(Intrinsic::ID id) {
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

void cleanupDeadCode(Module *M) {

  auto kernels = getKernels(M);
  auto reflects = getTagedFunctions(M, "pacxx.reflection", "");
  auto visitor = make_CallVisitor([](CallInst *) {});

  for (auto F : kernels)
    visitor.visit(F);
  for (auto F : reflects)
    visitor.visit(F);

  auto called = visitor.get();

  vector<Function *> deleted;
  for (auto &F : M->getFunctionList()) {

    if (find(kernels.begin(), kernels.end(), &F) == kernels.end() &&
        find(called.begin(), called.end(), &F) == called.end() &&
        find(reflects.begin(), reflects.end(), &F) == reflects.end() && 
        F.getName().find("__pacxx_block") == std::string::npos) {
      F.deleteBody();
      if (F.hasNUsesOrMore(1))
        F.replaceAllUsesWith(UndefValue::get(F.getType()));
      deleted.push_back(&F);
    }
  }

  for (auto F : deleted)
    F->eraseFromParent();

  vector<GlobalValue *> globals;
  for (auto &G : M->getGlobalList()) {
    bool hold = false;
    for (auto U : G.users()) {
      if (auto I = dyn_cast<Instruction>(U)) {
        if (I->getParent()) {
          auto F = I->getParent()->getParent();
          if (find(reflects.begin(), reflects.end(), F) != reflects.end()) {
            hold = true;
          }
        }
      }
      if (auto CE = dyn_cast<ConstantExpr>(U)) {
        for (auto CEU : CE->users()) {
          if (auto I = dyn_cast<Instruction>(CEU)) {
            if (I->getParent()) {
              auto F = I->getParent()->getParent();
              if (find(reflects.begin(), reflects.end(), F) != reflects.end()) {
                hold = true;
              }
            }
          }
        }
      }
    }
    if (hold)
      continue;
    G.removeDeadConstantUsers();

    bool isConstantMem = false; 
    bool isSharedMem = false;

    if (auto GV = dyn_cast<GlobalVariable>(&G)){
    // FIXME: is this realy true?
      if (GV->isConstant() && !GV->getMetadata("pacxx.as.constant") && GV->hasNUsesOrMore(1)){

        bool shouldBeConstant = false; 
        for (auto  U : GV->users()){
          if (isa<Instruction>(U)){
            shouldBeConstant = true; 
          }else if (auto CE = dyn_cast<ConstantExpr>(U)){
            for (auto  CU : CE->users()){
              if (isa<Instruction>(CU))
              shouldBeConstant = true; 
            }
          }
        }

        if (shouldBeConstant)
          GV->setMetadata("pacxx.as.constant", llvm::MDNode::get(GV->getContext(), nullptr));
      }
      isConstantMem = GV->getMetadata("pacxx.as.constant") != nullptr || GV->getType()->getAddressSpace() == 4;
      isSharedMem = GV->getMetadata("pacxx.as.shared") != nullptr || GV->getType()->getAddressSpace() == 3;
    }

    if (!isSharedMem && !isConstantMem)
      if (G.getType()->getAddressSpace() == 0 || G.hasNUses(0)) {
        G.replaceAllUsesWith(UndefValue::get(G.getType()));
        globals.push_back(&G);
      }
  }

  vector<GlobalAlias *> aliases;
  for (auto &A : M->getAliasList())
    aliases.push_back(&A);

  for (auto A : aliases) {
    auto Aliasee = A->getAliasee();
    if (isa<UndefValue>(Aliasee)) {
      A->replaceAllUsesWith(UndefValue::get(A->getType()));
      A->eraseFromParent();
      continue;
    }

    if (find(globals.begin(), globals.end(), A->getBaseObject()) !=
        globals.end()) {
      A->replaceAllUsesWith(UndefValue::get(A->getType()));
      A->eraseFromParent();
    }
  }

  for (auto G : globals)
    G->eraseFromParent();
}

bool isOpenCLFunction(StringRef name) {
  if (name.equals("_Z13get_global_idj"))
    return true;
  if (name.equals("_Z12get_group_idj"))
    return true;
  if (name.equals("_Z12get_local_idj"))
    return true;
  if (name.equals("_Z14get_local_sizej"))
    return true;

  return false;
}

void fixVectorOffset(Function *F, unsigned offset) {
  if (offset == 0 || !F)
    return;

  auto &ctx = F->getContext();

  for (auto &a : F->args()) {
    Instruction *I = &F->getBasicBlockList().front().getInstList().front();
    Argument *arg = &a;
    if (arg->getType()->isPointerTy() &&
        arg->getType()->getPointerElementType()->isStructTy()) {
      Type *arg_type = arg->getType()->getPointerElementType();
      if (arg_type->getStructName().startswith(STD_VECTOR_TYPE)) {
        Value *head =
            ConstantInt::get(Type::getInt64Ty(ctx), offset);

        auto bic1 =
            CastInst::Create(Instruction::BitCast, arg,
                             Type::getInt8Ty(ctx)->getPointerTo(
                                 arg->getType()->getPointerAddressSpace()),
                             "oldHead", I);
        auto newGEP =
            GetElementPtrInst::Create(bic1->getType()->getPointerElementType(),
                                      (Value *)bic1, head, "", I);
        auto bic2 = CastInst::Create(Instruction::BitCast, newGEP,
                                     arg->getType(), "newHead", I);
        SmallVector<Value *, 3> idx;
        idx.push_back(
            ConstantInt::get(Type::getInt64Ty(ctx), -1));
        idx.push_back(
            ConstantInt::get(Type::getInt64Ty(ctx), 0));
        idx.push_back(
            ConstantInt::get(Type::getInt64Ty(ctx), 0));
        vector<Instruction *> user;
        for (auto I = arg->user_begin(), E = arg->user_end(); I != E; ++I) {
          if (*I != bic1) {
            for (unsigned i = 0; i < I->getNumOperands(); ++i) {
              if (arg == I->getOperand(i)) {
                user.push_back(cast<Instruction>(*I));
              }
            }
          }
        }

        for (auto I : user) {
          for (unsigned i = 0; i < I->getNumOperands(); ++i) {
            if (arg == I->getOperand(i)) {
              I->setOperand(i, bic2);
            }
          }
        }
      }
    }
  }
}
}
