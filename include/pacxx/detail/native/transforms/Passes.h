//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

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