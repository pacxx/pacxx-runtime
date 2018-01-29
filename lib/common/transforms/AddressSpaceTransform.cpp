//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <cassert>
#include <iostream>
#include <set>
#include <sstream>
#include <vector>

#include "llvm/IR/Dominators.h"
#include <string>

#include "pacxx/detail/common/transforms/CallVisitor.h"
#include "pacxx/detail/common/transforms/ModuleHelper.h"

using namespace llvm;
using namespace std;
using namespace pacxx;

namespace {
template <typename T>
Instruction *getFirstInstructionForConstantExpr(T &kernels, ConstantExpr &CE) {
  for (auto CEU : CE.users()) {
    if (auto I = dyn_cast<Instruction>(CEU)) {
      if (I->getParent()) {
        return I;
      }
    }
    if (auto nextCE = dyn_cast<ConstantExpr>(CEU)) {
      return getFirstInstructionForConstantExpr(kernels, *nextCE);
    }
  }
  return nullptr;
}

template <typename T>
Function *getParentKernel(T &kernels, GlobalVariable &GV) {
  Function *F = nullptr;
  for (auto U : GV.users()) {
    Instruction *I = nullptr;
    if (isa<Instruction>(U)) {
      I = cast<Instruction>(U);
    }
    if (auto CE = dyn_cast<ConstantExpr>(U)) {
      I = getFirstInstructionForConstantExpr(kernels, *CE);
    }

    if (I && I->getParent()) {
      F = I->getParent()->getParent();
    }
    if (find(begin(kernels), end(kernels), F) != end(kernels)) {
      break;
    }
  }
  return F;
}

struct AddressSpaceTransform : public ModulePass {
  static char ID;
  AddressSpaceTransform() : ModulePass(ID) {}
  virtual ~AddressSpaceTransform() {}

  ValueToValueMapTy mapping;
  std::set<Value *> dead;

  void followUses(std::vector<Value *> &worklist, std::vector<Value *> &delayed,
                  unsigned AS = 0) {
    for (auto V : worklist) {
      if (auto ASC = dyn_cast<AddrSpaceCastInst>(V)) {
        Value *ptr = ASC->getOperand(0);
        mapping[ASC] = ptr;
        std::vector<Value *> directWorklist;
        for (auto U : V->users()) {
          if (auto GEP = dyn_cast<GetElementPtrInst>(U)) {
            std::vector<Value *> new_worklist;
            SmallVector<Value *, 3> idx;
            for (auto &I : GEP->indices())
              idx.push_back(I.get());
            auto newGEP = GetElementPtrInst::Create(
                ptr->getType()->getPointerElementType(), ptr, idx, "", GEP);
            mapping[GEP] = newGEP;
            dead.insert(GEP);
            for (auto GU : GEP->users())
              new_worklist.push_back(GU);
            followUses(new_worklist, delayed,
                       ptr->getType()->getPointerAddressSpace());
          } else
            directWorklist.push_back(U);
        }
        followUses(directWorklist, delayed);
      } else if (auto CI = dyn_cast<BitCastInst>(V)) {
        auto op0 = mapping[CI->getOperand(0)];
        auto newCI = new BitCastInst(
            op0,
            CI->getType()->getPointerElementType()->getPointerTo(
                op0->getType()->getPointerAddressSpace()),
            "", CI);
        mapping[CI] = newCI;
        dead.insert(CI);
        std::vector<Value *> new_worklist;
        for (auto U : CI->users())
          new_worklist.push_back(U);
        followUses(new_worklist, delayed, AS);
      } else if (auto LI = dyn_cast<LoadInst>(V)) {
        auto newLI = new LoadInst(mapping[LI->getOperand(0)], "", LI);
        newLI->setAlignment(4);
        newLI->setVolatile(LI->isVolatile());
        LI->replaceAllUsesWith(newLI);
        LI->eraseFromParent();
      } else if (auto SI = dyn_cast<StoreInst>(V)) {
        if (mapping[SI->getOperand(1)] != nullptr){ // operand not mapped do nothing 
        auto newSI =
            new StoreInst(SI->getOperand(0), mapping[SI->getOperand(1)],
                          SI->isVolatile(), SI);
        newSI->setAlignment(4);
        SI->eraseFromParent();
        }
      } else if (auto GEP = dyn_cast<GetElementPtrInst>(V)) {
        auto ptr = mapping[GEP->getOperand(0)];
        std::vector<Value *> new_worklist;
        SmallVector<Value *, 3> idx;
        for (auto &I : GEP->indices())
          idx.push_back(I.get());
        auto newGEP = GetElementPtrInst::Create(
            ptr->getType()->getPointerElementType(), ptr, idx, "", GEP);
        mapping[GEP] = newGEP;
        dead.insert(GEP);
        for (auto GU : GEP->users())
          new_worklist.push_back(GU);
        followUses(new_worklist, delayed, AS);
      } else if (auto PHI = dyn_cast<PHINode>(V)) {
        SmallVector<Value *, 8> mappedOps;
        bool delay = false;
        for (unsigned i = 0; i < PHI->getNumIncomingValues(); ++i) {
          if (mapping.find(PHI->getIncomingValue(i)) == mapping.end()) {
            delay = true;
            break;
          }
        }
        if (!delay) {
          // FIXME: we have to check first if all new values are in the same AS
          //        if not, we have to generate ASC Constant Expr for the
          //        incomming values and let the PHINode in the generic AS
          AS = mapping[PHI->getIncomingValue(0)]
                   ->getType()
                   ->getPointerAddressSpace();
          auto newPhi = PHINode::Create(
              PHI->getType()->getPointerElementType()->getPointerTo(AS),
              PHI->getNumIncomingValues(), "",
              PHI->getParent()->getFirstNonPHI());
          mapping[PHI] = newPhi;
          for (unsigned i = 0; i < PHI->getNumIncomingValues(); ++i) {
            // adding incomming values to set the new value
            newPhi->addIncoming(mapping[PHI->getIncomingValue(i)],
                                PHI->getIncomingBlock(i));
          }
          std::vector<Value *> new_worklist;
          dead.insert(PHI);
          for (auto U : PHI->users())
            new_worklist.push_back(U);
          followUses(new_worklist, delayed, AS);
        } else
          delayed.push_back(V);
      } else if (auto II = dyn_cast<IntrinsicInst>(V)) {
        SmallVector<Type *, 4> types;
        SmallVector<Value *, 4> args;
        for (unsigned i = 0; i < II->getNumOperands() - 1; ++i) {

          if (mapping.find(II->getOperand(i)) != mapping.end()) {
            types.push_back(mapping[II->getOperand(i)]->getType());
            args.push_back(mapping[II->getOperand(i)]);
          } else {
            types.push_back(II->getOperand(i)->getType());
            args.push_back(II->getOperand(i));
          }
        }
        auto Decl =
            Intrinsic::getDeclaration(II->getParent()->getParent()->getParent(),
                                      II->getIntrinsicID(), types);

        Decl = Intrinsic::remangleIntrinsicFunction(Decl).getValue();
        auto newII = CallInst::Create(Decl, args, "", II);
        mapping[II] = newII;
        dead.insert(II);
        std::vector<Value *> new_worklist;
        for (auto U : II->users())
          new_worklist.push_back(U);
        followUses(new_worklist, delayed, AS);
      } else if (auto SI = dyn_cast<SelectInst>(V)) {
        bool delay = mapping.find(SI->getTrueValue()) == mapping.end() ||
                     mapping.find(SI->getFalseValue()) == mapping.end();

        if (!delay) {
          auto newSelect = SelectInst::Create(
              SI->getCondition(), mapping[SI->getTrueValue()],
              mapping[SI->getFalseValue()], "", SI);
          mapping[SI] = newSelect;
          dead.insert(SI);
          std::vector<Value *> new_worklist;
          for (auto U : SI->users())
            new_worklist.push_back(U);
          followUses(new_worklist, delayed, AS);
        } else {
          delayed.push_back(V);
        }
      } else if (auto CI = dyn_cast<CallInst>(V)) {

        unsigned opi = 0;
        for (; opi < CI->getNumOperands(); ++opi)
          if (mapping.find(CI->getOperand(opi)) != mapping.end())
            break;

        auto ASC =
            new AddrSpaceCastInst(mapping[CI->getOperand(opi)],
                                  CI->getOperand(opi)->getType(), "", CI);
        CI->setOperand(opi, ASC);
      } else {
        llvm::errs() << "unhandled AS user\n";
        V->dump();
      }
    }
  }

  virtual bool runOnModule(Module &M) {
    bool modified = true;

    auto kernels = pacxx::getKernels(&M);

    auto visitor = make_CallVisitor([&](CallInst *I) {
      if (!I)
        return;

      if (!I->isInlineAsm()) {

        if (!isa<Function>(I->getCalledValue())) {
          I->dump();
        }
      }
    });

    //if (M.getTargetTriple().find("nvptx") != std::string::npos){
    // handle parameters to bring them into AS 1
    for (auto &F : kernels) {
      visitor.visit(F);

      // Mutate pointer types to bring them into AS 1
      auto &BB = F->getBasicBlockList().front();
      auto II = &BB.getInstList().front();
      bool mutate = false;
      for (auto &arg : F->args()) {
        if (arg.getType()->isPointerTy()) {
          if (arg.getType()->getPointerAddressSpace() == 0) {
            auto AL = new AllocaInst(arg.getType(), 0, "", II);
            auto SI = new StoreInst(&arg, AL, II);
            auto LI = new LoadInst(AL, "", II);
            arg.replaceAllUsesWith(LI);
            arg.mutateType(
                arg.getType()->getPointerElementType()->getPointerTo(1));

            auto ASC = new AddrSpaceCastInst(
                &arg, arg.getType()->getPointerElementType()->getPointerTo(0),
                "", II);
            LI->replaceAllUsesWith(ASC);
            LI->eraseFromParent();
            SI->eraseFromParent();
            AL->eraseFromParent();
            mutate = true;
          }
        }
      }

      if (mutate) {
        SmallVector<Type *, 8> Params;
        for (auto &arg : F->args())
          Params.push_back(arg.getType());

        Type *RetTy = F->getReturnType();
        FunctionType *NFTy = FunctionType::get(RetTy, Params, false);
        auto name = F->getName().str();
        F->setName("undead");
        auto NF = Function::Create(NFTy, F->getLinkage(), name, &M);
        auto DestI = NF->arg_begin();
        ValueToValueMapTy VMap;
        for (auto I = F->arg_begin(); I != F->arg_end(); ++I) {
          DestI->setName(I->getName());
          VMap[cast<Value>(I)] = cast<Value>(DestI++);
        }
        SmallVector<ReturnInst *, 8> returns;
        CloneFunctionInto(NF, F, VMap, true, returns);
        if (auto MD = M.getNamedMetadata("nvvm.annotations")) {
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

        // F->replaceAllUsesWith(NF);
        ReplaceUnsafe(F, NF);
        //  F->eraseFromParent();
      }
    }
    //}
    CEExtractor ceExtractor;

    kernels = pacxx::getKernels(&M);

    for (auto F : kernels) {
      ceExtractor.visit(F);
    }

    // handle shared memory declarations
    map<Function *, unsigned> SMMapping;
    SmallVector<GlobalVariable *, 4> deadGV;
    for (auto &GV : M.globals()) {
      if (GV.getMetadata("pacxx.as.shared") != nullptr &&
          GV.getType()->getPointerAddressSpace() != 3) {
        auto F = getParentKernel(kernels, GV);
        string newName = GV.getName().str() + ".sm";
        Type *elemType = GV.getType()->getPointerElementType();

        if (M.getTargetTriple().find("amdgcn") != std::string::npos){
          // For external shared memory on AMDGCN devices use the special function 
          // created by AMDGCNPrepair and replace the GV with the base pointer 
          // to SM
          if (GV.getType()->getElementType()->getArrayNumElements() == 0){ 
            IRBuilder<> builder(&F->front().front());
            auto VoidTy = Type::getInt8Ty(M.getContext());
            auto SMFTy = FunctionType::get(VoidTy->getPointerTo(3), false);
            auto CI = builder.CreateCall(M.getOrInsertFunction("__get_extern_shared_mem_ptr", SMFTy));
            auto BC = builder.CreateBitCast(CI, elemType->getPointerTo(3));
            auto ASC = builder.CreateAddrSpaceCast(BC, GV.getType());
            GV.replaceAllUsesWith(ASC);
            continue;
          }
        }

        auto newGV =
            new GlobalVariable(M, elemType, false,
                               llvm::GlobalValue::LinkageTypes::ExternalLinkage,
                               nullptr, // ConstantAggregateZero::get(elemType),
                               newName, &GV, GV.getThreadLocalMode(), 3, false);
        newGV->setAlignment(4);

        std::map<Instruction *, std::pair<Value *, Instruction *>> GVtoASC;
        for (auto U : GV.users()) {
          if (auto CE = dyn_cast<ConstantExpr>(U)) {
            if (CE->isCast()) {
              for (auto CU : CE->users()) {
                auto ASC =
                    new AddrSpaceCastInst(cast<Value>(newGV), GV.getType(), "",
                                          cast<Instruction>(CU));
                auto BC = new BitCastInst(ASC, CE->getType(), "",
                                          cast<Instruction>(CU));
                GVtoASC[cast<Instruction>(CU)] = make_pair(CE, BC);
              }
            } else if (CE->getOpcode() == Instruction::GetElementPtr) {
              for (auto CU : CE->users()) {
                if (isa<Instruction>(CU)) {
                  auto ASC =
                      new AddrSpaceCastInst(cast<Value>(newGV), GV.getType(),
                                            "", cast<Instruction>(CU));
                  SmallVector<Value *, 3> idx;
                  for (unsigned i = 1; i < CE->getNumOperands(); ++i) {
                    idx.push_back(CE->getOperand(i));
                  }
                  auto GEP = GetElementPtrInst::Create(
                      ASC->getType()->getPointerElementType(), ASC, idx, "",
                      cast<Instruction>(CU));
                  GVtoASC[cast<Instruction>(CU)] = make_pair(CE, GEP);
                } else {
                  llvm::errs() << "CU\n";
                  CU->dump();
                }
              }
            } else {
              llvm::errs() << "CE\n";
              CE->dump();
            }

          } else {
            auto ASC = new AddrSpaceCastInst(cast<Value>(newGV), GV.getType(),
                                             "", cast<Instruction>(U));
            GVtoASC[cast<Instruction>(U)] = make_pair(&GV, ASC);
          }
        }

        for (auto p : GVtoASC) {
          for (unsigned i = 0; i < p.first->getNumOperands(); ++i)
            if (p.first->getOperand(i) == p.second.first) {
              p.first->setOperand(i, p.second.second);
            }
        }
        deadGV.push_back(&GV);

        if (F) {
          unsigned i = SMMapping[F];
          newName = F->getName().str() + ".sm" + to_string(i);
          newGV->setName(newName);
          SMMapping[F] = i + 1;
        }
      }
      if (GV.getMetadata("pacxx.as.constant") &&
          !GV.getMetadata("pacxx.as.noopt")) {
        Type *oldType = GV.getType();
        if (GV.getType()->getPointerAddressSpace() == 0) {
          std::map<Instruction *, std::pair<Value *, Instruction *>> GVtoASC;
          unsigned AS = 4; 
          if (M.getTargetTriple().find("amdgcn") != std::string::npos)
            AS = 2;
          GV.mutateType(GV.getType()->getPointerElementType()->getPointerTo(AS));
          for (auto U : GV.users()) {
            auto ASC = new AddrSpaceCastInst(cast<Value>(&GV), oldType, "",
                                             cast<Instruction>(U));
            GVtoASC[cast<Instruction>(U)] = make_pair(&GV, ASC);
          }

          for (auto p : GVtoASC) {
            for (unsigned i = 0; i < p.first->getNumOperands(); ++i)
              if (p.first->getOperand(i) == p.second.first) {
                p.first->setOperand(i, p.second.second);
              }
          }
        }
      }
    }

    // test code for alloca in AS 3
    struct AllocaRewriter : public InstVisitor<AllocaRewriter> {
      std::vector<Value *> worklist;

      void visitAddrSpaceCastInst(AddrSpaceCastInst &ASC) {
        if (!ASC.getMetadata("pacxx.as.noopt"))
          worklist.push_back(&ASC);
      }
    } allocaRewriter;

    for (auto &F : kernels)
      allocaRewriter.visit(F);

    // delete old kernel functions
    cleanupDeadCode(&M);

    std::vector<Value *> delayed;
    followUses(allocaRewriter.worklist, delayed);
    do {
      std::vector<Value *> worklist(delayed);
      delayed.clear();
      followUses(worklist, delayed);
    } while (delayed.size());

    for (auto V : dead) {
      V->replaceAllUsesWith(UndefValue::get(V->getType()));
      cast<Instruction>(V)->eraseFromParent();
    }

    for (auto GV : deadGV) {
      GV->removeDeadConstantUsers();
      GV->replaceAllUsesWith(UndefValue::get(GV->getType()));
      GV->eraseFromParent();
    }

    return modified;
  }
};

char AddressSpaceTransform::ID = 0;
static RegisterPass<AddressSpaceTransform>
    X("pacxx-addr-space-transform", "Transforms generic AS to NVPTX AS", false,
      false);
} // namespace

namespace pacxx {
Pass *createAddressSpaceTransformPass() { return new AddressSpaceTransform(); }
} // namespace pacxx
