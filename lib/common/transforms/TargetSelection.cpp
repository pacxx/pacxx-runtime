//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "pacxx/detail/common/transforms/ModuleHelper.h"

#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "pacxx/detail/common/Log.h"

using namespace llvm;
using namespace pacxx;

namespace {
    class TargetSelection : public ModulePass {

    public:
        static char ID;

        TargetSelection(const SmallVector<std::string, 2>& targets) : ModulePass(ID), _targets(targets) {}

        virtual ~TargetSelection() {}

        virtual bool runOnModule(Module &M) {

            bool modified = false;
            auto kernels = pacxx::getTagedFunctions(&M, "nvvm.annotations", "kernel");
            for (auto kernel : kernels) {
                if (auto MD = kernel->getMetadata("pacxx.target")) {
                    auto Target = cast<MDString>(MD->getOperand(0).get());
                    if (!supportedTarget(Target)) {
                        kernel->eraseFromParent();
                        modified = true;
                    }
                }
            }
            return modified;
        }

    private:

        bool supportedTarget(const MDString * const Target) {
            bool supported = false;
            for(auto &target : _targets) {
               if(Target->getString().str() == target)
                   supported = true;
            }
            return supported;
        }

        const SmallVector<std::string, 2> _targets;
    };
}

char TargetSelection::ID = 0;

namespace pacxx {
Pass *createTargetSelectionPass(const SmallVector<std::string, 2>& targets) {
    return new TargetSelection(targets); }
}
