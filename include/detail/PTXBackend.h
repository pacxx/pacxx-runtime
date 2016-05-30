//
// Created by mhaidl on 29/05/16.
//

#ifndef PACXX_V2_PTXBACKEND_H
#define PACXX_V2_PTXBACKEND_H

#include <llvm/Target/TargetOptions.h>
#include <llvm/Target/TargetMachine.h>

#include "IRCompiler.h"

namespace llvm
{
  class Module;
  class Target;
}

namespace pacxx {
    namespace v2 {
        class PTXBackend : public IRCompiler {
        public:

            PTXBackend();
            virtual ~PTXBackend() { }

            virtual void initialize () override;
            virtual std::string compile(llvm::Module& M) override;

        private:
            const llvm::Target* _target;
            llvm::TargetOptions _options;
            std::unique_ptr<llvm::TargetMachine> _machine;
            std::string _cpu, _features;
        };
    }
}


#endif //PACXX_V2_PTXBACKEND_H
