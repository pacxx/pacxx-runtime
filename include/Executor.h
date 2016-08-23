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
#include <detail/cuda/CUDARuntime.h>
#include <detail/common/Exceptions.h>
#include <detail/IRRuntime.h>
#include <CodePolicy.h>
#include <detail/CoreInitializer.h>
#include <detail/common/Log.h>
#include <detail/cuda/PTXBackend.h>
#include <detail/MemoryManager.h>
#include <detail/KernelConfiguration.h>
#include <detail/KernelArgument.h>
#include <Promise.h>
#include <ModuleLoader.h>
#include <regex>
#include <cstdlib>

#ifdef __PACXX_V2_INTEROP
const char* llvm_start = nullptr;
int llvm_size = 0;
const char* reflection_start = nullptr;
int reflection_size = 0;
#else
extern const char llvm_start[];
extern int llvm_size;
extern const char reflection_start[];
extern int reflection_size;
#endif

namespace pacxx {
  namespace v2 {

    template<typename RuntimeT>
    class Executor {
    private:
      static bool _initialized;
    public:

      using CompilerT = typename RuntimeT::CompilerT;

      static auto& Create(bool load_internal = true) {// TODO: make dynamic fo different devices
        static Executor instance(0);

        if (!load_internal)
          _initialized = true;

        if (!_initialized) {
          ModuleLoader loader;
          auto M = loader.loadInternal(llvm_start, llvm_size);
          instance.setModule(std::move(M));
          _initialized = true;
        }
        return instance;
      }

      static auto& Create(std::string module_bytes) {// TODO: make dynamic fo different devices
        static Executor instance(0);

        if (!_initialized) {
          ModuleLoader loader;
          auto M = loader.loadInternal(module_bytes.data(), module_bytes.size());
          instance.setModule(std::move(M));
          _initialized = true;
        }
        return instance;
      }

    private:
      Executor(unsigned devID)
          : _runtime(std::make_unique<RuntimeT>(devID)), _mem_manager(*_runtime) {
        core::CoreInitializer::initialize();
      }

    public:

      void setModule(std::unique_ptr<llvm::Module> M);

      void setModule(std::string module_bytes);

      template<typename L, typename... Args>
      void run(const L& lambda, KernelConfiguration config, Args&& ... args) {
        auto& dev_lambda = _mem_manager.getTemporaryLambda(lambda);
        auto& K = get_kernel_by_name(typeid(L).name(), config, dev_lambda.get(), std::forward<Args>(args)...);
        K.launch();
      }

      template<typename L, typename CallbackFunc, typename... Args>
      void
      run_with_callback(const L& lambda, KernelConfiguration config, CallbackFunc&& cb, Args&& ... args) {
        auto& dev_lambda = _mem_manager.getTemporaryLambda(lambda);
        auto& K = get_kernel_by_name(typeid(L).name(), config, dev_lambda.get(), std::forward<Args>(args)...);
        K.setCallback(std::move(cb));
        K.launch();
      }


      template<typename... Args>
      auto& get_kernel_by_name(std::string name, KernelConfiguration config, Args&& ... args) {

        const llvm::Function* F = nullptr;
        const llvm::Module& M = _runtime->getModule();

        auto cleanName = [](const auto& name) {
          auto cleaned_name = std::regex_replace(name, std::regex("S[0-9A-Z]{0,9}_"), "");
          auto It = cleaned_name.find("$_");
          if (It == std::string::npos)
            return cleaned_name;
          It += 2;
          auto value = std::to_string(std::strtol(&cleaned_name[It], nullptr, 10)).size();
          cleaned_name.erase(It + value);
          return cleaned_name;
        };

        auto clean_name = cleanName(name);

//        auto pos = name.find("$_"); // get the lambdas unique id
//        if (pos != std::string::npos) {
//          pos += 2;
//          std::string numbers = "0123456789";
//          size_t first = name.find_first_of(numbers.c_str(), pos);
//          size_t last = first;
//          while (last != std::string::npos) {
//            size_t temp = last;
//            last = name.find_first_of(numbers.c_str(), last + 1);
//            if (last - temp > 1) {
//              last = temp;
//              break;
//            }
//          }
//
//          std::string id = name.substr(pos, last - first + 1);
//          for (const auto& func : M.functions()) {
//            if (func.getName().find(id) != llvm::StringRef::npos)
//              F = &func;
//          }
//        } else
//          F = M.getFunction(name);
        for (auto& Func : M.getFunctionList()) {
          auto fname = cleanName(Func.getName().str());

          if (fname.find(clean_name) != std::string::npos)
            F = &Func;
        }

        if (!F)
          throw common::generic_exception("Kernel function not found in module! " + name);

        size_t buffer_size = 0;
        std::vector<size_t> arg_offsets(F->arg_size());

        int offset = 0;

        std::transform(F->arg_begin(), F->arg_end(), arg_offsets.begin(), [&](const auto& arg) {
          auto arg_size = M.getDataLayout().getTypeAllocSize(arg.getType());
          auto arg_alignment =
              M.getDataLayout().getPrefTypeAlignment(arg.getType());

          auto arg_offset = (offset + arg_alignment - 1) & ~(arg_alignment - 1);
          offset = arg_offset + arg_size;
          buffer_size = offset;
          return arg_offset;
        });

        std::vector<char> args_buffer(buffer_size);
        auto ptr = args_buffer.data();
        size_t i = 0;
        common::for_each_in_arg_pack([&](auto&& arg) {
          auto offset = arg_offsets[i++];
          meta::memory_translation mtl;
          auto targ = mtl(_mem_manager, arg);
          std::memcpy(ptr + offset, &targ, sizeof(decltype(targ)));
        }, std::forward<Args>(args)...);

        auto& K = _runtime->getKernel(F->getName().str());
        K.configurate(config);
        K.setArguments(args_buffer);
        return K;
      }

      template<typename... Args>
      void run_interop(std::string name, KernelConfiguration config, const std::vector<KernelArgument>& args) {

        const llvm::Module& M = _runtime->getModule();
        const llvm::Function* F = M.getFunction(name);

        if (!F)
          throw common::generic_exception("Kernel function not found in module! " + name);

        size_t buffer_size = 0;
        std::vector<size_t> arg_offsets(F->arg_size());

        int offset = 0;

        std::transform(F->arg_begin(), F->arg_end(), arg_offsets.begin(), [&](const auto& arg) {
          auto arg_size = M.getDataLayout().getTypeAllocSize(arg.getType());
          auto arg_alignment =
              M.getDataLayout().getPrefTypeAlignment(arg.getType());

          auto arg_offset = (offset + arg_alignment - 1) & ~(arg_alignment - 1);
          offset = arg_offset + arg_size;
          buffer_size = offset;
          return arg_offset;
        });

        std::vector<char> args_buffer(buffer_size);
        auto ptr = args_buffer.data();
        size_t i = 0;

        for (const auto& arg : args) {
          auto offset = arg_offsets[i++];
          std::memcpy(ptr + offset, arg.address, arg.size);
        }

        auto& K = _runtime->getKernel(F->getName().str());
        K.configurate(config);
        K.setArguments(args_buffer);
        K.launch();
      }

      template<typename T>
      DeviceBuffer<T>& allocate(size_t count) {
        return *_runtime->template allocateMemory<T>(count);
      }

      RawDeviceBuffer& allocateRaw(size_t bytes) {
        return *_runtime->allocateRawMemory(bytes);
      }

      template<typename T>
      void free(DeviceBuffer<T>& buffer) {
        _runtime->template deleteMemory(&buffer);
      }

      void freeRaw(RawDeviceBuffer& buffer) {
        _runtime->deleteRawMemory(&buffer);
      }

      auto& mm() { return _mem_manager; }

      auto& rt() { return *_runtime; }

      void synchronize() { _runtime->synchronize(); }

      auto& getPassManager() { return _runtime->getPassManager(); }

      template<typename PromisedTy, typename... Ts>
      auto& getPromise(Ts&& ... args) {
        auto promise = new BindingPromise<PromisedTy>(std::forward<Ts>(args)...);
        return *promise;
      };

      template<typename PromisedTy>
      void forgetPromise(BindingPromise<PromisedTy>& instance) {

        delete &instance;

      };


    private:
      std::unique_ptr<RuntimeT> _runtime;
      MemoryManager _mem_manager;
    };

    template<typename T>
    bool Executor<T>::_initialized = false;

    template<typename T>
    void Executor<T>::setModule(std::unique_ptr<llvm::Module> M) {
      _runtime->link(std::move(M));
    }

    template<typename T>
    void Executor<T>::setModule(std::string module_bytes) {
      ModuleLoader loader;
      auto M = loader.loadInternal(module_bytes.data(), module_bytes.size());
      Create().setModule(std::move(M));
    }

    using RuntimeT = CUDARuntime;

    template<typename T = RuntimeT>
    auto& get_executor() { return Executor<RuntimeT>::Create(); }

  }
}

#endif // PACXX_V2_EXECUTOR_H
