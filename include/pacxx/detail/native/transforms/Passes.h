#pragma once
namespace llvm {
class Pass;
}

namespace pacxx {
llvm::Pass *createPACXXNativeBarrierPass();
llvm::Pass *createPACXXNativeLinkerPass();
llvm::Pass *createPACXXSelectEmitterPass();
llvm::Pass *createPACXXNativeSMPass();
llvm::Pass *createSPMDVectorizerPass();
}