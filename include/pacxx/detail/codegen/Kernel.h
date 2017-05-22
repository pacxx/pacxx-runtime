//
// Created by m_haid02 on 19.05.17.
//

#pragma once

#include "pacxx/Executor.h"
#include "pacxx/detail/KernelConfiguration.h"
#include "pacxx/detail/common/Log.h"
#include "pacxx/detail/common/Meta.h"
#include "pacxx/detail/device/DeviceCode.h"
#include "pacxx/detail/device/DeviceFunctionDecls.h"
#include <cstddef>
#include <regex>
#include <type_traits>
#include <utility>
#include <type_traits>
#include <experimental/type_traits>

namespace pacxx {
namespace v2 {
enum class Target {
  Generic,
  GPU,
  CPU
};

//
// THIS IS THE GENERIC KERNEL EXPRESSION
//

template<typename L>
[[pacxx::kernel]] [[pacxx::target("Generic")]] void genericKernel(L callable) noexcept {
  callable();
}

template<typename L>
[[pacxx::kernel]] [[pacxx::target("GPU")]] void genericKernelGPU(L callable) noexcept {
  callable();
}

template<typename L>
[[pacxx::kernel]] [[pacxx::target("CPU")]] void genericKernelCPU(L callable) noexcept {
  callable();
}

template<Target T> struct kernel_caller {
private:
  template<typename L>
  static void launch(const L &F) {

    switch (T) {
    case Target::Generic:genericKernel(F);
      break;
    case Target::CPU:genericKernelCPU(F);
      break;
    case Target::GPU:genericKernelGPU(F);
      break;
    }
  }

public:
  template<typename L>
  static void call(const L &F, const KernelConfiguration &config) {
    auto &executor = Executor::get(config.executor);

// In PACXX V1 the compiler removed the kernel_call in the host code
// this however, was highly unstable between the two compilation passes and
// overly complicated.
// In PACXX V2 we use the __device_code__ macro to distinguish between the two
// compilation phases
// This approach has the advantage that we do not have to break up type safety.
#ifdef __device_code__
    launch(F);
#else
    executor.run(F, config);
#endif
  }

  template<typename L, typename CallbackTy>
  static void call_with_cb(const L &F, const KernelConfiguration &config,
                           CallbackTy &callback) {
    auto &executor = Executor::get(config.executor);

#ifdef __device_code__
    launch(F);
#else
    executor.run_with_callback(F, config, callback);
#endif
  }
};

template<typename L, Target targ> class _kernel {
public:
  _kernel(L lambda, KernelConfiguration config)
      : _function(std::forward<L>(lambda)), _config(config) {}

  void operator()() const {
    using caller = kernel_caller<targ>;
    caller::call(_function, _config);
  }

  auto synchronize() { __message("synchronize not implemented yet"); }

private:
  L _function;
  KernelConfiguration _config;
};

template<typename L, typename CB, Target targ>
class _kernel_with_cb {
public:
  _kernel_with_cb(L &&lambda, KernelConfiguration config, CB &&callback)
      : _function(std::forward<L>(lambda)), _config(config), _callback(std::forward<CB>(callback)) {}

  void operator()() {
    using caller = kernel_caller<targ>;
    caller::call_with_cb(_function, _config, _callback);
  }

  auto synchronize() { __message("synchronize not implemented yet"); }

private:
  L _function;
  KernelConfiguration _config;
  CB _callback;
};

template<typename Func, Target targ = Target::Generic>
auto kernel(Func &lambda, KernelConfiguration config) {
  return _kernel<Func, targ>(std::forward<Func>(lambda), config);
};

template<typename Func, typename CallbackFunc, Target targ = Target::Generic>
auto kernel_with_cb(Func &lambda, KernelConfiguration config, CallbackFunc &&CB) {
  return _kernel_with_cb<Func, CallbackFunc, targ>(
      std::forward<Func>(lambda), config, std::forward<CallbackFunc>(CB));
};
}
}