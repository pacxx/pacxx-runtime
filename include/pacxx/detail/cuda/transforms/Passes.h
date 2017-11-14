//===-----------------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
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
llvm::Pass *createNVPTXPrepairPass();
llvm::Pass *createAddressSpaceTransformPass();
}