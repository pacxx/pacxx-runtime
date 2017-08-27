#pragma once

#include "llvm/Pass.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Constants.h"
#include <vector>
#include <set>
#include <map>

namespace llvm {
Pass *createPACXXReflectionRemoverPass();
Pass *createPACXXSpirPass();
Pass *createPACXXNvvmPass();
Pass *createPACXXNvvmRegPass(bool runtime = false);
Pass *createPACXXSPIRVectorFixerPass(size_t offset = 0);
Pass *createPACXXTargetSelectPass(const SmallVector<std::string, 2>&);

// native backend passes
Pass *createSPMDVectorizerPass();
Pass *createPACXXLivenessAnalyzerPass();
Pass *createPACXXNativeBarrierPass();
Pass *createPACXXNativeSMPass();
Pass *createPACXXNativeLinkerPass();
Pass *createPACXXIntrinsicSchedulerPass();
Pass *createPACXXSelectEmitterPass();
Pass *createPACXXIntrinsicMapperPass();
}
