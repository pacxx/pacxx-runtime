//
// Created by mhaidl on 29/05/16.
//

#ifndef PACXX_V2_EXECUTOR_H
#define PACXX_V2_EXECUTOR_H

#include <cuda.h>
#include <detail/CUDARuntime.h>
#include <detail/Exceptions.h>
#include <detail/IRRuntime.h>
#include <llvm/IR/Module.h>
#include <memory>
#include <string>

#include "CodePolicy.h"
#include "detail/CoreInitializer.h"
#include "detail/IRCompiler.h"
#include "detail/Log.h"
#include "detail/PTXBackend.h"

namespace pacxx {
namespace v2 {
class Executor {
public:
  template <typename CompilerT, typename RuntimeT>
  Executor(std::unique_ptr<llvm::Module> M,
           CodePolicy<CompilerT, RuntimeT> &&policy)
      : _M(std::move(M)), _compiler(std::make_unique<CompilerT>()),
        _runtime(std::make_unique<RuntimeT>(0)) {// TODO: make dynamic fo different devices
    core::CoreInitializer::initialize();

    _compiler->initialize();
    _runtime->linkMC(_compiler->compile(*_M));
  }

  template <typename... Args> void run(std::string name, Args &&... args) {

    auto F = _M->getFunction(name);

    if (!F)
      throw common::generic_exception("Kernel function not found in module!");

    size_t buffer_size = 0;
    std::vector<size_t> arg_sizes;
    for (const auto &arg : F->args()) {
      auto arg_size = _M->getDataLayout().getTypeAllocSize(arg.getType());
      auto arg_alignment =
          _M->getDataLayout().getPrefTypeAlignment(arg.getType());
      if (arg_size <= arg_alignment)
        buffer_size += arg_alignment;
      else
        buffer_size +=
            arg_size * (static_cast<size_t>(arg_size / arg_alignment) + 1);

      arg_sizes.push_back(arg_size);
    }

    std::vector<char> args_buffer(buffer_size);
    auto ptr = args_buffer.data();
    size_t i = 0;
    common::for_each_in_arg_pack([&](auto &&arg) {
      auto size = arg_sizes[i++];
      std::memcpy(ptr, &arg, size);
      ptr += size;
    }, std::forward<Args>(args)...);
    
    _runtime->setArguments(args_buffer);
  }

private:
  std::unique_ptr<llvm::Module> _M;
  std::unique_ptr<IRCompiler> _compiler;
  std::unique_ptr<IRRuntime> _runtime;
};
}
}

#endif // PACXX_V2_EXECUTOR_H
