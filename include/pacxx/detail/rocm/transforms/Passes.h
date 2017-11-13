//
// Created by m_haid02 on 29.08.17.
//

#pragma once

namespace llvm {
class Pass;
}

namespace pacxx {
llvm::Pass *createAMDGCNPrepairPass();
}