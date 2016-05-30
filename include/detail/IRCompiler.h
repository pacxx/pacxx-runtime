//
// Created by mhaidl on 29/05/16.
//

#ifndef PACXX_V2_IRCOMPILER_H
#define PACXX_V2_IRCOMPILER_H

#include <string>

namespace llvm
{
    class Module;
}

namespace pacxx
{
    namespace v2
    {
        class IRCompiler
        {
        public:
            virtual void initialize() = 0;
            virtual std::string compile(llvm::Module& M) = 0;
        };
    }
}



#endif //PACXX_V2_IRCOMPILER_H
