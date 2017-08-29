//
// Created by m_haid02 on 29.08.17.
//

#pragma once

#include <llvm/ADT/SmallVector.h>

namespace llvm {
class Pass;
}

namespace pacxx {
llvm::Pass *createPACXXIntrinsicSchedulerPass();
llvm::Pass *createPACXXNvvmRegPass(bool);
llvm::Pass *createPACXXIntrinsicMapperPass();
llvm::Pass *createPACXXReflectionPass();
llvm::Pass *createPACXXReflectionCleanerPass();
llvm::Pass *createPACXXReflectionRemoverPass();
llvm::Pass *createPACXXTargetSelectPass(const llvm::SmallVector<std::string, 2>& targets);
}