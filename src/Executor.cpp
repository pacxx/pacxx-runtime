//
// Created by mhaidl on 04/06/16.
//

#include "Executor.h"

namespace pacxx
{
namespace v2
{
  bool Executor::_initialized = false;

  Executor& Executor::Create(){
    auto& instance = Create(CodePolicy<PTXBackend, CUDARuntime>());
    if (!_initialized) {
      ModuleLoader loader;
      auto M = loader.loadInternal(llvm_start, llvm_size);
      instance.setModule(std::move(M));
      _initialized = true;
    }
    return instance;
  }

  void Executor::setModule(std::unique_ptr<llvm::Module> M) {
    _M = std::move(M);
    _runtime->linkMC(_compiler->compile(*_M));
  }
}
}