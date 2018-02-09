//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef PACXX_V2_EXECUTOR_H
#define PACXX_V2_EXECUTOR_H

#include "Promise.h"
#include "pacxx/detail/CoreInitializer.h"
#include "pacxx/detail/DeviceBuffer.h"
#include "pacxx/detail/KernelArgument.h"
#include "pacxx/detail/KernelConfiguration.h"
#include "pacxx/detail/Runtime.h"
#include "pacxx/detail/common/Exceptions.h"
#include "pacxx/detail/common/Log.h"
#include "pacxx/detail/common/TearDown.h"
#include "pacxx/pacxx_config.h"
#ifdef PACXX_ENABLE_CUDA
#include "pacxx/detail/cuda/CUDARuntime.h"
#endif
#ifdef PACXX_ENABLE_HIP
#include "pacxx/detail/rocm/HIPRuntime.h"
#endif
#include "pacxx/detail/Event.h"
#include "pacxx/detail/codegen/Kernel.h"
#include "pacxx/detail/native/NativeRuntime.h"
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <regex>
#include <string>

#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>

namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

// set the default backend
#ifndef PACXX_ENABLE_CUDA
using Runtime = pacxx::v2::NativeRuntime;
#else
using Runtime = pacxx::v2::CUDARuntime;
#endif

namespace pacxx {
namespace v2 {

class Executor;

enum ExecutingDevice { GPUNvidia, CPU, GPUAMD };

Executor &get_executor(unsigned id = 0);
template <typename T = Runtime, typename... Ts>
Executor &get_executor(Ts... args);

class Executor {
public:
  friend void initializeModule(Executor &exec);
  friend void initializeModule(Executor &exec, const char *ptr, size_t size);

  static std::vector<Executor> &getExecutors();

  static Executor &get(unsigned id = 0) {
    auto &executors = getExecutors();
    if (executors.empty()) {
      CreateDefault();
    }
    return executors[id];
  }

private:
  static Executor &CreateDefault() {
    int runtime = 0;
    auto str = common::GetEnv("PACXX_DEFAULT_RT");
    if (str != "") {
      runtime = std::stoi(str);
    }

    switch (runtime) {
    case 2: // HIP Runtime
#ifdef PACXX_ENABLE_HIP
      try {
        return Create<HIPRuntime>(0);
      } catch (...) {
        __verbose("No ROCm Device found: Using Fallback to CUDARuntime for "
                  "GPU execution as default Executor");
      }
#else
      __verbose("HIP Runtime not linked in! Falling back to CUDARuntime!");
#endif
    case 0: // CUDA Runtime
#ifdef PACXX_ENABLE_CUDA
      try {
        return Create<CUDARuntime>(0);
      } catch (...) {
        __verbose("No CUDA Device found: Using Fallback to NativeRuntime for "
                  "CPU execution as default Executor");
      }
#else
      __verbose("CUDA Runtime not linked in! Falling back to NativeRuntime!");
#endif
    case 1: // Native Runtime
    default:
      break;
    }
    return Create<NativeRuntime>(0);
  }

public:
  template <typename T = Runtime, typename... Ts>
  static Executor &Create(Ts... args) {
    std::unique_ptr<Runtime> rt(new T(args...));
    if (!rt->checkSupportedHardware())
      throw common::generic_exception("no supported device available!");

    auto &executors = getExecutors();

    executors.emplace_back(std::move(rt));
    auto &instance = executors.back();

    instance._id = executors.size() - 1;
    __verbose("Created new Executor with id: ", instance.getID());
    try {
      initializeModule(instance);
    } catch (common::generic_exception ex) {
      __exception(ex.what());
      __error("Module initialization failed!");
    }
    return instance;
  }
  Executor(std::unique_ptr<Runtime> &&rt);

  Executor(Executor &&other);

  ~Executor();

private:
  std::string cleanName(const std::string &name);

public:
  unsigned getID();

  template <typename L, pacxx::v2::Target targ = pacxx::v2::Target::Generic>
  void launch(L callable, KernelConfiguration config) {
    auto kern = pacxx::v2::codegenKernel<L, targ>(callable);
    run(kern.name, callable, config);
  }

  template <typename L, pacxx::v2::Target targ = pacxx::v2::Target::Generic>
  auto launch(L callable, KernelConfiguration config,
              std::promise<void> &promise) {
    auto kern = pacxx::v2::codegenKernel<L, targ>(callable);
    auto future = promise.get_future();
    run_with_callback(kern.name, callable, config,
                      [&]() mutable { promise.set_value(); });
    return future;
  }

  template <typename L, pacxx::v2::Target targ = pacxx::v2::Target::Generic,
            typename CB>
  void launch_with_callback(L callable, KernelConfiguration config,
                            CB &&callback) {
    auto kern = pacxx::v2::codegenKernel<L, targ>(callable);
    run_with_callback(kern.name, callable, config, callback);
  }

  template <typename T> auto getVectorizationWidth() {
    return _runtime->getPreferedVectorSize(sizeof(T));
  }

  size_t getConcurrentCores();

  ExecutingDevice getExecutingDeviceType();

  void run_by_name(std::string name, const void *args, size_t size,
                   KernelConfiguration config) {
    // auto& dev_lambda = _mem_manager.getTemporaryLambda(lambda);
    auto &K = get_kernel_by_name(name, config, args, size);
    K.launch();
  }

private:
  void setModule(std::unique_ptr<llvm::Module> M);

  template <typename L>
  void run(std::string name, const L &lambda, KernelConfiguration config) {
    // auto& dev_lambda = _mem_manager.getTemporaryLambda(lambda);
    auto &K = get_kernel_by_name(name, config, lambda);
    enshadowArgs();
    K.profile();
    K.launch();
  }

  template <typename L, typename CallbackFunc, typename... Args>
  void run_with_callback(std::string name, const L &lambda,
                         KernelConfiguration config, CallbackFunc &&cb,
                         Args &&... args) {
    auto &K =
        get_kernel_by_name(name, config, lambda, std::forward<Args>(args)...);
    enshadowArgs();
    K.profile();
    K.setCallback(std::move(cb));
    K.launch();
  }

  template <typename L>
  Kernel &get_kernel_by_name(std::string name, KernelConfiguration config,
                             const L &lambda) {
    return get_kernel_by_name(name, config, &lambda, sizeof(L));
  }

  Kernel &get_kernel_by_name(std::string name, KernelConfiguration config,
                             const void *args, size_t size);

public:
  template <typename T>
  DeviceBuffer<T> &allocate(size_t count,
                            MemAllocMode mode = MemAllocMode::Standard) {
    __verbose("allocating memory: ", sizeof(T) * count);

    if (mode == MemAllocMode::Unified)
      if (!_runtime->supportsUnifiedAddressing())
        throw std::bad_alloc();

    return *_runtime->allocateMemory<T>(count, mode);
  }

  template <typename T> void free(T *ptr) {
    _runtime->template deleteMemory(_runtime->template translateMemory(ptr));
  }

  template <typename T> void free(DeviceBuffer<T> &buffer) {
    _runtime->template deleteMemory(&buffer);
  }

  bool supportsDoublePrecission() {
    return _runtime->isSupportingDoublePrecission();
  }

  Runtime &rt();

  void synchronize();

  template <typename PromisedTy, typename... Ts>
  auto &getPromise(Ts &&... args) {
    auto promise = new BindingPromise<PromisedTy>(std::forward<Ts>(args)...);
    return *promise;
  }

  template <typename PromisedTy>
  void forgetPromise(BindingPromise<PromisedTy> &instance) {
    delete &instance;
  }

  llvm::LLVMContext &getLLVMContext() { return *_ctx; }

  void enshadowArgs() {
    _runtime->enshadowMemory();
    __verbose("Args enshadowed");
  }

  void restoreArgs() {
    _runtime->restoreMemory();
    __verbose("Args restored");
  }

  std::unique_ptr<Event> createEvent() { return _runtime->createEvent(); }

private:
  std::unique_ptr<llvm::LLVMContext> _ctx;
  std::unique_ptr<Runtime> _runtime;
  unsigned _id;
};

} // namespace v2
} // namespace pacxx

#endif // PACXX_V2_EXECUTOR_H
