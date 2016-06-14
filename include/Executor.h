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
  template <typename RuntimeT>
class Executor {
private:
  static bool _initialized;
public:

  using CompilerT = typename RuntimeT::CompilerT;

   static auto& Create() {// TODO: make dynamic fo different devices
     static Executor instance (0);

     if (!_initialized) {
       ModuleLoader loader;
       auto M = loader.loadInternal(llvm_start, llvm_size);
       instance.setModule(std::move(M));
       _initialized = true;
     }
    return instance;
  }

private:
  Executor(unsigned devID)
      :  _compiler(std::make_unique<CompilerT>()),
        _runtime(std::make_unique<RuntimeT>(devID)), _mem_manager(*_runtime) {
    core::CoreInitializer::initialize();
    _compiler->initialize();
  }

public:

  void setModule(std::unique_ptr<llvm::Module> M);
  template <typename L, typename... Args> void run(const L& lambda, KernelConfiguration config, Args &&... args) {
    auto& dev_lambda = _mem_manager.getTemporaryLambda(lambda);
    run_by_name(typeid(L).name(), config, dev_lambda.get(), std::forward<Args>(args)...);
  }

  template <typename... Args> void run_by_name(std::string name, KernelConfiguration config, Args &&... args) {

    const Function *F = nullptr;

    auto pos = name.find("$_"); // get the lambdas unique id
    if (pos != std::string::npos) {
      pos += 2;
      std::string numbers = "0123456789";
      size_t first = name.find_first_of(numbers.c_str(), pos);
      size_t last = first;
      while (last != std::string::npos) {
        size_t temp = last;
        last = name.find_first_of(numbers.c_str(), last+1);
        if (last - temp > 1) {
          last = temp;
          break;
        }
      }

      std::string id = name.substr(pos, last - first + 1);
      for (const auto &func : _M->functions()) {
        if (func.getName().find(id) != llvm::StringRef::npos)
          F = &func;
      }
    }
    else
      F = _M->getFunction(name);

    if (!F)
      throw common::generic_exception("Kernel function not found in module! " + name);

    size_t buffer_size = 0;
    std::vector<size_t> arg_offsets(F->arg_size());

    int offset = 0;

    std::transform(F->arg_begin(), F->arg_end(), arg_offsets.begin(), [&](const auto& arg){
      auto arg_size = _M->getDataLayout().getTypeAllocSize(arg.getType());
      auto arg_alignment =
          _M->getDataLayout().getPrefTypeAlignment(arg.getType());

      /*if (arg_size <= arg_alignment)
        buffer_size += arg_alignment;
      else
        buffer_size +=
            arg_size * (static_cast<size_t>(arg_size / arg_alignment) + 1);*/
      auto arg_offset = (offset + arg_alignment - 1) & ~(arg_alignment - 1);
      offset = arg_offset + arg_size;
      buffer_size = offset;
      return arg_offset;
    });

    std::vector<char> args_buffer(buffer_size);
    auto ptr = args_buffer.data();
    size_t i = 0;
    common::for_each_in_arg_pack([&](auto &&arg) {
      auto offset = arg_offsets[i++];
      meta::memory_translation mtl;
      auto targ = mtl(_mem_manager, arg);
      std::memcpy(ptr + offset, &targ, sizeof(decltype(targ)));
    }, std::forward<Args>(args)...);

    auto& K = _runtime->getKernel(F->getName().str());
    K.configurate(config);
    K.setArguments(args_buffer);
    K.launch();
  }

  template <typename T> DeviceBuffer<T>& allocate(size_t count){
    return *_runtime->template allocateMemory<T>(count);
  }

  RawDeviceBuffer& allocateRaw(size_t bytes){
    return *_runtime->allocateRawMemory(bytes);
  }

  auto& mm(){ return _mem_manager; }

private:
  std::unique_ptr<llvm::Module> _M;
  std::unique_ptr<CompilerT> _compiler;
  std::unique_ptr<RuntimeT> _runtime;
  MemoryManager _mem_manager;
};

  template <typename T>
  bool Executor<T>::_initialized = false;

  template <typename T>
  void Executor<T>::setModule(std::unique_ptr<llvm::Module> M) {
    _M = std::move(M);
    _runtime->linkMC(_compiler->compile(*_M));
  }


}
}

#endif // PACXX_V2_EXECUTOR_H
