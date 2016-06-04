//
// Created by mhaidl on 29/05/16.
//

#ifndef PACXX_V2_EXECUTOR_H
#define PACXX_V2_EXECUTOR_H

#include <cuda.h>

#include <llvm/IR/Module.h>
#include <memory>
#include <string>
#include <algorithm>
#include <detail/DeviceBuffer.h>

#include "detail/cuda/CUDARuntime.h"
#include "detail/common/Exceptions.h"
#include "detail/IRRuntime.h"
#include "CodePolicy.h"
#include "detail/CoreInitializer.h"
#include "detail/IRCompiler.h"
#include "detail/common/Log.h"
#include "detail/cuda/PTXBackend.h"
#include "detail/MemoryManager.h"
#include "detail/KernelConfiguration.h"
#include "ModuleLoader.h"

extern const char llvm_start[];
extern int llvm_size;
extern const char reflection_start[];
extern int reflection_size;


namespace pacxx {
namespace v2 {
class Executor {
private:
  static bool _initialized;
public:

  static Executor& Create();

   template <typename CompilerT, typename RuntimeT>
   static auto& Create(
           CodePolicy<CompilerT, RuntimeT> &&policy) {// TODO: make dynamic fo different devices
    __verbose("Creating Executor for ", typeid(RuntimeT).name());
    static Executor executor(std::forward<CodePolicy<CompilerT, RuntimeT>>(policy), 0);
    return executor;
  }

private:
  template <typename CompilerT, typename RuntimeT>
  Executor(CodePolicy<CompilerT, RuntimeT> &&policy, unsigned devID)
      :  _compiler(std::make_unique<CompilerT>()),
        _runtime(std::make_unique<RuntimeT>(devID)), _mem_manager(*_runtime) {
    core::CoreInitializer::initialize();
    _compiler->initialize();
  }

public:

  void setModule(std::unique_ptr<llvm::Module> M);

  template <typename... Args> void run(std::string name, KernelConfiguration config, Args &&... args) {

    __message(name);
    const Function* F = nullptr;

    for (const auto& func : _M->functions())
    {
      __message(func.getName().str());
      if (func.getName().find(name) != llvm::StringRef::npos)
        F = &func;
    }

    if (!F)
      throw common::generic_exception("Kernel function not found in module!");

    size_t buffer_size = 0;
    std::vector<size_t> arg_sizes(F->arg_size());

    std::transform(F->arg_begin(), F->arg_end(), arg_sizes.begin(), [&](const auto& arg){
      auto arg_size = _M->getDataLayout().getTypeAllocSize(arg.getType());
      auto arg_alignment =
          _M->getDataLayout().getPrefTypeAlignment(arg.getType());
      if (arg_size <= arg_alignment)
        buffer_size += arg_alignment;
      else
        buffer_size +=
            arg_size * (static_cast<size_t>(arg_size / arg_alignment) + 1);

      return arg_size;
    });

    std::vector<char> args_buffer(buffer_size);
    auto ptr = args_buffer.data();
    size_t i = 0;
    common::for_each_in_arg_pack([&](auto &&arg) {
      auto size = arg_sizes[i++];
      meta::memory_translation mtl;
      auto targ = mtl(_mem_manager, arg);
      std::memcpy(ptr, &targ, size);
      ptr += size;
    }, std::forward<Args>(args)...);

    auto& K = _runtime->getKernel(F->getName().str());
    K.configurate(config);
    K.setArguments(args_buffer);
    K.launch();
  }

  template <typename T> DeviceBuffer<T>& allocate(size_t count){
    DeviceBufferBase* ptr = _runtime->allocateMemory(count * sizeof(T));
    return *static_cast<DeviceBuffer<T>*>(ptr);
  }

  RawDeviceBuffer& allocateRaw(size_t bytes){
    return *_runtime->allocateRawMemory(bytes);
  }

  auto& mm(){ return _mem_manager; }

private:
  std::unique_ptr<llvm::Module> _M;
  std::unique_ptr<IRCompiler> _compiler;
  std::unique_ptr<IRRuntime> _runtime;
  MemoryManager _mem_manager;
};
}
}

#endif // PACXX_V2_EXECUTOR_H
