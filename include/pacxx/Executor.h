//
// Created by mhaidl on 29/05/16.
//

#ifndef PACXX_V2_EXECUTOR_H
#define PACXX_V2_EXECUTOR_H

#include "pacxx/pacxx_config.h"
#include "CodePolicy.h"
#include "ModuleLoader.h"
#include "Promise.h"
#include "pacxx/detail/CoreInitializer.h"
#include "pacxx/detail/DeviceBuffer.h"
#include "pacxx/detail/IRRuntime.h"
#include "pacxx/detail/KernelArgument.h"
#include "pacxx/detail/KernelConfiguration.h"
#include "pacxx/detail/common/Exceptions.h"
#include "pacxx/detail/common/TearDown.h"
#include "pacxx/detail/common/Log.h"
#ifdef PACXX_ENABLE_CUDA
#include "pacxx/detail/cuda/CUDARuntime.h"
#endif
#include "pacxx/detail/native/NativeRuntime.h"
#include "pacxx/detail/codegen/Kernel.h"
#include "pacxx/detail/Event.h"
#include "pacxx/detail/cuda/CUDAEvent.h"  // TODO: move event create to the runtimes
#include "pacxx/detail/native/NativeEvent.h"
#include <algorithm>
#include <cstdlib>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Module.h>
#include <memory>
#include <regex>
#include <string>

#ifdef __PACXX_RUNTIME_LINKING
const char *llvm_start = nullptr;
const char *llvm_end = nullptr;
#else
extern const char llvm_start[];
extern const char llvm_end[];
#endif

#ifndef PACXX_ENABLE_CUDA
using Runtime = pacxx::v2::NativeRuntime;
#else
using Runtime = pacxx::v2::CUDARuntime;
#endif

namespace pacxx {
namespace v2 {

class Executor;

enum ExecutingDevice {
  GPUNvidia,
  CPU
};

Executor &get_executor(unsigned id = 0);
template<typename T = Runtime, typename... Ts> Executor &get_executor(Ts... args);

class Executor {
public:

  static auto &getExecutors() {
    static std::vector<Executor> *executors = new std::vector<Executor>();
    return *executors; // TODO: free resources at application's exit
  }

  static Executor &get(unsigned id = 0) {
    auto &executors = getExecutors();
    if (executors.empty()) {

#ifdef PACXX_ENABLE_CUDA
      if (CUDARuntime::checkSupportedHardware()) {
        Create<CUDARuntime>(0); // TODO: make dynamic for different devices
      }
      else {
        __verbose("No CUDA Device found: Using Fallback to NativeRuntime for CPU execution as default Executor");
#endif
        Create<NativeRuntime>(0);
#ifdef PACXX_ENABLE_CUDA
      }
#endif
    }
    return executors[id];
  }

  template<typename T = Runtime, typename... Ts> static Executor &Create(Ts... args) {
    std::unique_ptr<IRRuntime> rt(new T(args...));
    return Executor::Create(std::move(rt));
  }

  static Executor &Create(std::unique_ptr<IRRuntime> rt, std::string module_bytes = "") {
    auto &executors = getExecutors();

    executors.emplace_back(std::move(rt));
    auto &instance = executors.back();

    instance._id = executors.size() - 1;
    __verbose("Created new Executor with id: ", instance.getID());
    ModuleLoader loader(instance.getLLVMContext());
    if (module_bytes == "") {
      auto M = loader.loadInternal(llvm_start, llvm_end - llvm_start);
      instance.setModule(std::move(M));
    } else {
      ModuleLoader loader(instance.getLLVMContext());
      auto M = loader.loadInternal(module_bytes.data(), module_bytes.size());
      instance.setModule(std::move(M));
    }

    return instance;
  }

  Executor(std::unique_ptr<IRRuntime> &&rt);

  Executor(Executor &&other);

  ~Executor(){}

private:
  std::string cleanName(const std::string &name);

public:

  unsigned getID();

  template<typename L, pacxx::v2::Target targ = pacxx::v2::Target::Generic>
  void launch(L callable, KernelConfiguration config)
  {
    pacxx::v2::codegenKernel<L, targ>(callable);
    run(callable, config);
  }

  template<typename L, pacxx::v2::Target targ = pacxx::v2::Target::Generic, typename CB>
  void launch_with_callback(L callable, KernelConfiguration config, CB &&callback)
  {
    pacxx::v2::codegenKernel<L, targ>(callable);
    run_with_callback(callable, config, callback);
  }

  template<typename T>
  auto getVectorizationWidth() {
    return _runtime->getPreferedVectorSize(sizeof(T));
  }

  size_t getConcurrentCores();

  ExecutingDevice getExecutingDeviceType();

private:
  void setModule(std::unique_ptr<llvm::Module> M);

  void setModule(std::string module_bytes);

  template<typename L>
  void run(const L &lambda, KernelConfiguration config) {
    // auto& dev_lambda = _mem_manager.getTemporaryLambda(lambda);
    auto &K = get_kernel_by_name(typeid(L).name(), config,
                                 lambda);
    K.launch();
  }

  template<typename L, typename CallbackFunc, typename... Args>
  void run_with_callback(const L &lambda, KernelConfiguration config,
                         CallbackFunc &&cb, Args &&... args) {
    auto &K = get_kernel_by_name(typeid(L).name(), config,
                                 lambda,
                                 std::forward<Args>(args)...);
    K.setCallback(std::move(cb));
    K.launch();
  }

  template<typename L>
  auto &get_kernel_by_name(std::string name, KernelConfiguration config,
                           const L &lambda) {

    std::string FName;
    const llvm::Module &M = _runtime->getModule();
    auto it = _kernel_translation.find(name);
    if (it == _kernel_translation.end()) {
      auto clean_name = cleanName(name);
      for (auto &p : _kernel_translation)
        if (p.first.find(clean_name) != std::string::npos) {
          FName = p.second;
          //_kernel_translation[name] = F.getName().str();
        }
    } else
      FName = it->second;

    auto F = M.getFunction(FName);
    if (!F) {
      throw common::generic_exception("Kernel function not found in module! " +
          cleanName(name));
    }

    auto &K = _runtime->getKernel(FName);
    // K.setName(FName);
    K.configurate(config);
    K.setLambdaPtr(&lambda);

    _runtime->evaluateStagedFunctions(K);

    return K;
  }

public:
  template<typename T>
  DeviceBuffer<T> &allocate(size_t count, T *host_ptr = nullptr, MemAllocMode mode = MemAllocMode::Standard) {
    __verbose("allocating memory: ", sizeof(T) * count);

    if (mode == MemAllocMode::Unified)
      __verbose("Runtime supports unified addressing: ", _runtime->supportsUnifiedAddressing());

    switch (_runtime->getKind()) {
#ifdef PACXX_ENABLE_CUDA
    case IRRuntime::RuntimeKind::RK_CUDA:
      return *cast<CUDARuntime>(_runtime.get())->template allocateMemory(count,
                                                                  host_ptr, mode);
#endif
    case IRRuntime::RuntimeKind::RK_Native:
      return *cast<NativeRuntime>(_runtime.get())->template allocateMemory(count,
                                                                  host_ptr, mode);
    }

    throw pacxx::common::generic_exception("unreachable code");
  }

  RawDeviceBuffer &allocateRaw(size_t bytes, MemAllocMode mode = MemAllocMode::Standard);

  template<typename T> void free(DeviceBuffer<T> &buffer) {
    switch (_runtime->getKind()) {
#ifdef PACXX_ENABLE_CUDA
    case IRRuntime::RuntimeKind::RK_CUDA: cast<CUDARuntime>(_runtime.get())->template deleteMemory(&buffer);
      break;
#endif
    case IRRuntime::RuntimeKind::RK_Native: cast<NativeRuntime>(_runtime.get())->template deleteMemory(&buffer);
      break;
    default:llvm_unreachable("this runtime type is not defined!");
    }
  }

  void freeRaw(RawDeviceBuffer &buffer);

  bool supportsDoublePrecission(){ return _runtime->isSupportingDoublePrecission(); }

  IRRuntime &rt();

  void synchronize();

  llvm::legacy::PassManager &getPassManager();

  template<typename PromisedTy, typename... Ts>
  auto &getPromise(Ts &&... args) {
    auto promise = new BindingPromise<PromisedTy>(std::forward<Ts>(args)...);
    return *promise;
  }

  template<typename PromisedTy>
  void forgetPromise(BindingPromise<PromisedTy> &instance) {

    delete &instance;
  }

  LLVMContext &getLLVMContext() { return *_ctx; }

  Event& createEvent(){
    _events.emplace_back();
    auto& event = _events.back();
    switch (_runtime->getKind()) {
#ifdef PACXX_ENABLE_CUDA
    case IRRuntime::RuntimeKind::RK_CUDA:
      event.reset(new CUDAEvent());
      break;
#endif
    case IRRuntime::RuntimeKind::RK_Native:
      event.reset(new NativeEvent());
      break;
    }
    return *event;
  }


private:
  std::unique_ptr<LLVMContext> _ctx;
  std::unique_ptr<IRRuntime> _runtime;
  std::map<std::string, std::string> _kernel_translation;
  std::vector<std::unique_ptr<Event>> _events;
  unsigned _id;
};

}
}

#endif // PACXX_V2_EXECUTOR_H
