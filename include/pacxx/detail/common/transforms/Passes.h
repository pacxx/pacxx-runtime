//
// Created by m_haid02 on 29.08.17.
//

#pragma once

#include <llvm/ADT/SmallVector.h>

namespace llvm {
class Pass;
}

namespace pacxx {
llvm::Pass *createIntrinsicSchedulerPass();
llvm::Pass *createMemoryCoalescingPass(bool);
llvm::Pass *createIntrinsicMapperPass();
llvm::Pass *createMSPGenerationPass();
llvm::Pass *createMSPCleanupPass();
llvm::Pass *createMSPRemoverPass();
llvm::Pass *createTargetSelectionPass(const llvm::SmallVector<std::string, 2>& targets);
llvm::Pass *createPACXXCodeGenPrepare();
}