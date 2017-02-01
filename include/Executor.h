//
// Created by mhaidl on 29/05/16.
//

#ifndef PACXX_V2_EXECUTOR_H
#define PACXX_V2_EXECUTOR_H

#include <cuda.h>

#include <llvm/IR/Module.h>
#include <llvm/IR/Constants.h>
#include <memory>
#include <string>
#include <algorithm>
#include <detail/DeviceBuffer.h>
#include <detail/cuda/CUDARuntime.h>
#include <detail/native/NativeRuntime.h>
#include <detail/common/Exceptions.h>
#include <detail/IRRuntime.h>
#include <CodePolicy.h>
#include <detail/CoreInitializer.h>
#include <detail/common/Log.h>
#include <detail/cuda/PTXBackend.h>
#include <detail/native/NativeBackend.h>
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

#ifdef __PACXX_V2_NATIVE_RUNTIME
  using Runtime = pacxx::v2::NativeRuntime;
#else
  using Runtime = pacxx::v2::CUDARuntime;
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
          ModuleLoader loader(instance.getLLVMContext());
          auto M = loader.loadInternal(llvm_start, llvm_size);
          instance.setModule(std::move(M));
          instance.setMSPModule(loader.loadInternal(reflection_start, reflection_size));
          _initialized = true;
        }
        return instance;
      }

      static auto& Create(std::string module_bytes) {// TODO: make dynamic fo different devices
        static Executor instance(0);

        if (!_initialized) {
          ModuleLoader loader(instance.getLLVMContext());
          auto M = loader.loadInternal(module_bytes.data(), module_bytes.size());
          instance.setModule(std::move(M));
          _initialized = true;
        }
        return instance;
      }

    private:
      Executor(unsigned devID)
          : _ctx(),  _runtime(std::make_unique<RuntimeT>(devID)), _mem_manager(*_runtime) {
            core::CoreInitializer::initialize();
      }

      std::string cleanName(const std::string& name) {
        auto cleaned_name = std::regex_replace(name, std::regex("S[0-9A-Z]{0,9}_"), "");
        cleaned_name = std::regex_replace(cleaned_name, std::regex("5pacxx"), ""); // bad hack
        cleaned_name = std::regex_replace(cleaned_name, std::regex("2v2"), ""); // bad hack
        // cleaned_name = std::regex_replace(cleaned_name, std::regex("S[0-9A-Z]{0,9}_"), "");
        auto It = cleaned_name.find("$_");
        if (It == std::string::npos)
          return cleaned_name;
        It += 2;
        auto value = std::to_string(std::strtol(&cleaned_name[It], nullptr, 10)).size();
        cleaned_name.erase(It + value);
        return cleaned_name;
      }

    public:

      void setMSPModule(std::unique_ptr<llvm::Module> M) {
        _runtime->initializeMSP(std::move(M));
      }

      void setModule(std::unique_ptr<llvm::Module> M);

      void setModule(std::string module_bytes);

      template<typename L, typename... Args>
      void run(const L& lambda, KernelConfiguration config, Args&& ... args) {
        // auto& dev_lambda = _mem_manager.getTemporaryLambda(lambda);
        auto& K = get_kernel_by_name(typeid(L).name(), config, std::forward<const L>(lambda),
                                     std::forward<Args>(args)...);
        K.launch();
      }

      template<typename L, typename CallbackFunc, typename... Args>
      void
      run_with_callback(const L& lambda, KernelConfiguration config, CallbackFunc&& cb, Args&& ... args) {
        auto& K = get_kernel_by_name(typeid(L).name(), config, std::forward<const L>(lambda),
                                     std::forward<Args>(args)...);
        K.setCallback(std::move(cb));
        K.launch();
      }


      template<typename... Args>
      auto& get_kernel_by_name(std::string name, KernelConfiguration config, Args&& ... args) {

        std::string FName;
        const llvm::Module& M = _runtime->getModule();
        auto it = _kernel_translation.find(name);
        if (it == _kernel_translation.end()) {
          auto clean_name = cleanName(name);
          for (auto& p : _kernel_translation)
            if (p.first.find(clean_name) != std::string::npos) {
              FName = p.second;
              //_kernel_translation[name] = F.getName().str();
            }
        }
        else
          FName = it->second;

        auto F = M.getFunction(FName); 
        if (!F) {
          throw common::generic_exception("Kernel function not found in module! " + cleanName(name));
        }

        auto& K = _runtime->getKernel(FName);
        K.setName(FName);
        K.configurate(config);

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
        __verbose("host arg buffer size is: ", K.getHostArgumentsSize());
        std::vector<char> host_args_buffer(K.getHostArgumentsSize());

        auto ptr = args_buffer.data();
        auto hptr = host_args_buffer.data();
        size_t i = 0;
        ptrdiff_t hoffset = 0;
        void* lambdaPtr = nullptr;

        common::for_first_in_arg_pack([&](auto& lambda) {
          lambdaPtr = (void*) &lambda;
        }, std::forward<Args>(args)...);


        common::for_each_in_arg_pack([&](auto&& arg) {
          if (i == 0) {
            if (host_args_buffer.size() > 0)
              std::memcpy(hptr + hoffset, &lambdaPtr, sizeof(decltype(lambdaPtr)));
            hoffset += sizeof(decltype(lambdaPtr));
          } else if (host_args_buffer.size() > 0) {
            std::memcpy(hptr + hoffset, &arg, sizeof(decltype(arg)));
            hoffset += sizeof(decltype(arg));
          }

          auto offset = arg_offsets[i++];
          auto targ = meta::memory_translation{}(_mem_manager, arg);
//          __warning(sizeof(decltype(arg)), " ", hoffset, " ", typeid(arg).name());
          std::memcpy(ptr + offset, &targ, sizeof(decltype(targ)));
        }, std::forward<Args>(args)...);


        K.setHostArguments(host_args_buffer);
        K.setArguments(args_buffer);

        _runtime->evaluateStagedFunctions(K);

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
      DeviceBuffer<T>& allocate(size_t count, T *host_ptr = nullptr) {
        return *_runtime->template allocateMemory<T>(count, host_ptr);
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

      auto& getLLVMContext() { return _ctx; }

    private:
      LLVMContext _ctx;
      std::unique_ptr<RuntimeT> _runtime;
      MemoryManager _mem_manager;
      std::map<std::string, std::string> _kernel_translation;
    };

    template<typename T>
    bool Executor<T>::_initialized = false;

    template<typename T>
    void Executor<T>::setModule(std::unique_ptr<llvm::Module> M) {

      _runtime->link(std::move(M));

      auto& nM = _runtime->getModule();
      for (auto& F : nM.getFunctionList())
        _kernel_translation[cleanName(F.getName().str())] = F.getName().str();
    }

    template<typename T>
    void Executor<T>::setModule(std::string module_bytes) {
      ModuleLoader loader(_ctx);
      auto M = loader.loadInternal(module_bytes.data(), module_bytes.size());
      Create().setModule(std::move(M));
    }


    template<typename T = Runtime>
    auto& get_executor() { return Executor<T>::Create(); }

  }
}

#endif // PACXX_V2_EXECUTOR_H
