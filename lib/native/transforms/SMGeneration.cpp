//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SMGeneration.h"

using namespace llvm;
using namespace pacxx;

SMGeneration::SMGeneration() : ModulePass(ID) {
    __verbose("created sm pass\n");
}

SMGeneration::~SMGeneration() {}

void SMGeneration::releaseMemory() {}

void SMGeneration::getAnalysisUsage(AnalysisUsage &AU) const {}

bool SMGeneration::runOnModule(Module &M) {

    __verbose("Generating shared memory \n");
    auto kernels = pacxx::getKernels(&M);
    for(auto &kernel : kernels) {
        runOnKernel(kernel);
    }

    return true;
}

void SMGeneration::runOnKernel(Function *kernel) {
    auto argIt = kernel->arg_end();

    // the sm_size is always the second last arg
    Value *sm_size = &*(--(--argIt));

    createSharedMemoryBuffer(kernel, sm_size);
}

void SMGeneration::createSharedMemoryBuffer(Function *func, Value *sm_size) {

    Module *M = func->getParent();

    auto internal_sm = getSMGlobalsUsedByKernel(M, func, true);
    auto external_sm = getSMGlobalsUsedByKernel(M, func, false);

    if(!internal_sm.empty() || !external_sm.empty()) {
        BasicBlock *entry = &func->front();
        BasicBlock *sharedMemBB = BasicBlock::Create(func->getContext(), "shared mem", func, entry);

        if (!internal_sm.empty()) {
            __verbose("internal shared memory found\n");
            createInternalSharedMemoryBuffer(*M, func, internal_sm, sharedMemBB);
        }

        if (!external_sm.empty()) {
            __verbose("external shared memory found\n");
            createExternalSharedMemoryBuffer(*M, func, external_sm, sm_size, sharedMemBB);
        }

        BranchInst::Create(entry, sharedMemBB);

        __verbose("created shared memory");
    }
}

set<GlobalVariable *> SMGeneration::getSMGlobalsUsedByKernel(Module *M, Function *func, bool internal) {
    set<GlobalVariable *> sm;
    for (auto &GV : M->globals()) {
        bool consider = false;
       if((GV.hasMetadata() && GV.getMetadata("pacxx.as.shared"))) {
            Type *sm_type = GV.getType()->getElementType();
            consider = internal ? sm_type->getArrayNumElements() != 0 : sm_type->getArrayNumElements() == 0;
        }
        if(consider) {
            for (User *GVUsers : GV.users()) {
                if (Instruction *Inst = dyn_cast<Instruction>(GVUsers)) {
                    if (Inst->getParent()->getParent() == func) {
                        sm.insert(&GV);
                    }
                }

                if(ConstantExpr *constExpr = dyn_cast<ConstantExpr>(GVUsers)) {
                    vector<ConstantUser> smUsers = findInstruction(func, constExpr);
                    set<Constant *> constantsToRemove;
                    for(auto &smUser : smUsers) {
                        auto inst = smUser._inst;
                        for(auto constant : smUser._constants) {
                            Instruction *constInst = constant->getAsInstruction();
                            constInst->insertBefore(inst);
                            inst->replaceUsesOfWith(constant, constInst);
                            inst = constInst;
                            constantsToRemove.insert(constant);
                        }
                    }

                    for(auto constant : constantsToRemove) {
                        constant->dropAllReferences();
                    }

                    sm.insert(&GV);
                }
            }
        }
    }
    __verbose("found #sm ", sm.size());
    return sm;
}

vector<SMGeneration::ConstantUser> SMGeneration::findInstruction(Function *func, ConstantExpr * constExpr) {
    vector<ConstantUser> smUsers;
    for (auto &B : *func) {
        for (auto &I : B) {
            vector<ConstantExpr *> constants;
            vector<ConstantExpr *> tmp;
            Instruction *inst = &I;
            for (auto &op : inst->operands()) {
                if (ConstantExpr *opConstant = dyn_cast<ConstantExpr>(op.get())) {
                    bool usesSM = false;
                    if (opConstant == constExpr) {
                        constants.push_back(opConstant);
                        smUsers.push_back(ConstantUser(inst, constants));
                    }
                    else {
                        tmp.push_back(opConstant);
                        lookAtConstantOps(opConstant, constExpr, tmp, constants, &usesSM);
                        if (usesSM) {
                            smUsers.push_back(ConstantUser(inst, constants));
                        }
                    }
                }
            }
        }
    }
    return smUsers;
}

void SMGeneration::lookAtConstantOps(ConstantExpr *constExp, ConstantExpr *smUser,
                                                 vector<ConstantExpr *> &tmp,
                                                 vector<ConstantExpr *> &constants,
                                                 bool *usesSM) {
   for(auto &op : constExp->operands()) {
       if(ConstantExpr *opConstant = dyn_cast<ConstantExpr>(op.get())) {
           if(opConstant == smUser) {
               for(auto constant : tmp) {
                   constants.push_back(constant);
               }
               constants.push_back(opConstant);
               *usesSM = true;
               return;
           }

           tmp.push_back(opConstant);
           lookAtConstantOps(opConstant, smUser, tmp, constants, usesSM);
       }
   }
}

void SMGeneration::createInternalSharedMemoryBuffer(Module &M,
                                                                Function *kernel,
                                                                set<GlobalVariable *> &globals,
                                                                BasicBlock *sharedMemBB) {

    const DataLayout &dl = M.getDataLayout();

    for (auto GV : globals) {

        Type *sm_type = GV->getType()->getElementType();
        IRBuilder<> builder(sharedMemBB);

        auto sm_alloc = builder.CreateAlloca(sm_type);
        sm_alloc->setAlignment(dl.getPrefTypeAlignment(sm_type));
        auto cast = builder.CreateBitCast(sm_alloc, sm_type->getPointerTo(0));
       // if (GV->hasInitializer() && !isa<UndefValue>(GV->getInitializer()))
       //     new StoreInst(GV->getInitializer(), sm_alloc, sharedMemBB);

        replaceAllUsesInKernel(kernel, GV, cast);
    }
}

void SMGeneration::createExternalSharedMemoryBuffer(Module &M,
                                                                Function *kernel,
                                                                set<GlobalVariable *> &globals,
                                                                Value *sm_size,
                                                                BasicBlock *sharedMemBB) {
    for (auto GV : globals) {
        Type *GVType = GV->getType()->getElementType();
        Type *sm_type = nullptr;
        if (GVType->getArrayElementType()->isSingleValueType()) {
            unsigned vectorWidth = kernel->hasFnAttribute("simd-size") ?
                                   stoi(kernel->getFnAttribute("simd-size").getValueAsString().str()) : 1;

            sm_type = VectorType::get(GVType->getArrayElementType(), vectorWidth);
        }
        else
            sm_type = GVType->getArrayElementType();

        Value *typeSize =
            ConstantInt::get(Type::getInt32Ty(M.getContext()), M.getDataLayout().getTypeAllocSize(sm_type));

        //calc number of elements
        BinaryOperator *div = BinaryOperator::CreateUDiv(sm_size, typeSize, "numElem", sharedMemBB);
        Value *sm_alloc = new AllocaInst(sm_type, 0, div,
                                              "external_sm", sharedMemBB);
        cast<AllocaInst>(sm_alloc)->setAlignment(M.getDataLayout().getPrefTypeAlignment(sm_type));
        if (sm_alloc->getType() != GV->getType())
            sm_alloc = new BitCastInst(sm_alloc, GV->getType(), "cast", sharedMemBB);

        replaceAllUsesInKernel(kernel, GV, sm_alloc);
    }
}

void SMGeneration::replaceAllUsesInKernel(Function *kernel, Value *from, Value *with) {
    auto UI = from->use_begin(), E = from->use_end();
    for (; UI != E;) {
        Use &U = *UI;
        ++UI;
        auto *Usr = dyn_cast<Instruction>(U.getUser());
        if (Usr && Usr->getParent()->getParent() == kernel)
            U.set(with);
    }
    return;
}

char SMGeneration::ID = 0;

namespace pacxx {
    llvm::Pass* createSMGenerationPass() { return new SMGeneration(); }
}


