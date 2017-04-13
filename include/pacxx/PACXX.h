//
// Created by mhaidl on 04/06/16.
//

#ifndef PACXX_V2_PACXX_H
#define PACXX_V2_PACXX_H

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

template<class U, U> struct IntegralConstant {};

template<class U, U *u> std::string mangledSymbolName() {
  std::string null = typeid(IntegralConstant<U *, nullptr>).name();
  std::string symbol = typeid(IntegralConstant<U *, u>).name();
  return symbol.substr(null.size() - 3, symbol.size() - null.size() + 0);
}

namespace pacxx {
namespace v2 {

enum class Target{
  Generic,
  GPU,
  CPU
};

template<class T, int size, class Enable = void> struct vec {};

template<class T, int size>
struct vec<T, size, typename std::enable_if<(size & (size - 1)) == 0>::type> {
  typedef T type __attribute__((ext_vector_type(size)));
};

template<typename F, typename... Args>
[[pacxx::reflect]] auto _stage(F func, Args &&... args) {
  return static_cast<long>(func(std::forward<Args>(args)...));
}

template<typename Arg> auto stage(const Arg &val) {
  return static_cast<Arg>(_stage([&] { return val; }));
}

template<typename T, size_t __sm_size = 0> class shared_memory {
public:
  template<typename U = T,
      typename std::enable_if<!std::is_void<U>::value &&
          __sm_size == 0>::type * = nullptr>
  shared_memory() {
#ifdef __device_code__
    [[pacxx::shared]] extern T ptr[];
#else
    T *ptr = nullptr;
#endif
    sm_ptr = reinterpret_cast<decltype(sm_ptr)>(ptr);
  }

  template<typename U = T,
      typename std::enable_if<!std::is_void<U>::value &&
          __sm_size != 0>::type * = nullptr>
  shared_memory() {
    [[pacxx::shared]] T ptr[__sm_size];
    sm_ptr = reinterpret_cast<decltype(sm_ptr)>(ptr);
  }

private:
  T /*__attribute__((address_space(3)))*/ *sm_ptr;

public:
  T &operator[](int idx) { return sm_ptr[idx]; }

  const T &operator[](int idx) const { return sm_ptr[idx]; }
};

template<
    typename T,
    typename std::enable_if<!std::is_pointer<
        typename std::remove_reference<T>::type>::value>::type * = nullptr>
T forward_sec(T val) {
  return val;
}

template<
    typename T,
    typename std::enable_if<std::is_pointer<
        typename std::remove_reference<T>::type>::value>::type * = nullptr>
typename std::remove_reference<T>::type forward_sec(T &val) {
  return val;
}

//
// THIS IS THE GENERIC KERNEL EXPRESSION
//

template<size_t _C, typename L, typename... ArgTys>
[[pacxx::kernel]] [[pacxx::target("Generic")]] void genericKernel(L callable, ArgTys... args) noexcept {
  callable(args...);
}

template<size_t _C, typename L, typename... ArgTys>
[[pacxx::kernel]] [[pacxx::target("GPU")]] void genericKernelGPU(L callable, ArgTys... args) noexcept {
  callable(args...);
}

template<size_t _C, typename L, typename... ArgTys>
[[pacxx::kernel]] [[pacxx::target("CPU")]] void genericKernelCPU(L callable, ArgTys... args) noexcept {
  callable(args...);
}

template<Target T, size_t _C> struct kernel_caller {
private:
  template<typename L, typename... Ts>
  static void launch(const L &F, Ts &&... args){
    switch(T)
    {
    case Target::Generic:
      genericKernel<_C, L,
                    meta::add_gpu_reference_t<std::remove_reference_t<Ts>>...>(
          F, args...);
      break;
    case Target::CPU:
      genericKernelCPU<_C, L,
                       meta::add_gpu_reference_t<std::remove_reference_t<Ts>>...>(
          F, args...);
      break;
    case Target::GPU:
      genericKernelGPU<_C, L,
                       meta::add_gpu_reference_t<std::remove_reference_t<Ts>>...>(
          F, args...);
      break;
    }
  }

public:
  template<typename L, typename... Ts>
  static void call(const L &F, const KernelConfiguration &config,
                   Ts &&... args) {
    auto &executor = Executor::get(config.executor);

// In PACXX V1 the compiler removed the kernel_call in the host code
// this however, was highly unstable between the two compilation passes and
// overly complicated.
// In PACXX V2 we use the __device_code__ macro to distinguish between the two
// compilation phases
// This approach has the advantage that we do not have to break up type safety.
#ifdef __device_code__
    launch(F, std::forward<Ts>(args)...);
#else
    __verbose("launching kernel: ", typeid(F).name(), " with executor: ", config.executor);
    executor.run(F, config, std::forward<Ts>(args)...);
#endif
  }

  template<typename L, typename CallbackTy, typename... Ts>
  static void call_with_cb(const L &F, const KernelConfiguration &config,
                           CallbackTy &callback, Ts &&... args) {
    auto &executor = Executor::get(config.executor);

#ifdef __device_code__
    launch(F, std::forward<Ts>(args)...);
#else
    __verbose("launching kernel with callback: ", typeid(F).name(), " with executor: ", config.executor);
    executor.run_with_callback(F, config, callback, std::forward<Ts>(args)...);
#endif
  }
};

template<typename L, Target targ, size_t _C> class _kernel {
public:
  _kernel(const L &lambda, KernelConfiguration config)
      : _function(lambda), _config(config) {}

  template<typename... Ts> void operator()(Ts &&... args) const {
    // using caller = kernel_caller<_C,
    // decltype(&std::remove_const_t<std::remove_reference_t<L>>::operator())>;
    using caller = kernel_caller<targ, _C>;

    caller::call(_function, _config, std::forward<Ts>(args)...);
  }

  auto synchronize() { __message("synchronize not implemented yet"); }

private:
  const L &_function;
  KernelConfiguration _config;
};

template<typename L, typename CB, Target targ, size_t _C>
class _kernel_with_cb {
public:
  _kernel_with_cb(const L &lambda, KernelConfiguration config, CB &&callback)
      : _function(lambda), _config(config), _callback(std::move(callback)) {}

  template<typename... Ts> void operator()(Ts &&... args) {
    using caller = kernel_caller<targ, _C>;

    caller::call_with_cb(_function, _config, _callback,
                         std::forward<Ts>(args)...);
  }

  auto synchronize() { __message("synchronize not implemented yet"); }

private:
  meta::callable_wrapper<L> _function;
  KernelConfiguration _config;
  CB _callback;
};

template<typename Func, Target targ = Target::Generic,
    size_t versioning = __COUNTER__>
auto kernel(const Func &lambda, KernelConfiguration config) {
  return _kernel<decltype(lambda), targ, versioning>(lambda, config);
};

template<typename Func, typename CallbackFunc, Target targ = Target::Generic,
    size_t versioning = __COUNTER__>
auto kernel_with_cb(const Func &lambda, KernelConfiguration config,
                    CallbackFunc &&CB) {
  return _kernel_with_cb<decltype(lambda), CallbackFunc, targ, versioning>(
      lambda, config, std::forward<CallbackFunc>(CB));
};

}
}
#endif // PACXX_V2_KERNEL_H_H
