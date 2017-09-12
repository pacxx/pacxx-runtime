#pragma once
namespace llvm {
class Pass;
}

namespace pacxx {
llvm::Pass *createBarrierGenerationPass();
llvm::Pass *createKernelLinkerPass();
llvm::Pass *createMaskedMemTransformPass();
llvm::Pass *createSMGenerationPass();
llvm::Pass *createSPMDVectorizerPass();
}