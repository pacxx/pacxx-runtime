//
// Created by mhaidl on 29/05/16.
//

#ifndef PACXX_V2_EXECUTOR_H
#define PACXX_V2_EXECUTOR_H

#include <memory>
#include <string>
#include <llvm/IR/Module.h>

#include "detail/IRCompiler.h"
#include "detail/CoreInitializer.h"
#include "detail/PTXBackend.h"
#include "detail/Log.h"

namespace pacxx{
  namespace v2{
    class Executor
    {
    public:

      Executor()
      {
        core::CoreInitializer::initialize();
      }

      template <typename... Args>
          void run(std::unique_ptr<llvm::Module>& M, std::string name, Args&&... args)
      {
        if (!_compiler) {
          _compiler.reset(new PTXBackend()); // TODO: make dynamic for different backends
          _compiler->initialize();
        }

        auto MC = _compiler->compile(M);
        __message(MC);
      }

    private:
      std::unique_ptr<IRCompiler> _compiler;
    };
  }
}


#endif //PACXX_V2_EXECUTOR_H
