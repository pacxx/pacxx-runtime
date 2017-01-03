//
// Created by mhaidl on 04/06/16.
//

#ifndef PACXX_V2_KERNEL_H_H
#define PACXX_V2_KERNEL_H_H

#include <type_traits>
#include <utility>
#include <cstddef>
#include "detail/common/Log.h"
#include "detail/common/Timing.h"
#include "detail/KernelConfiguration.h"
#include "Executor.h"
#include "detail/device/DeviceCode.h"
#include "detail/common/Meta.h"
#include "detail/device/DeviceFunctionDecls.h"
#include <regex>

template<class U, U>
struct IntegralConstant {
};

template<class U, U* u>
std::string mangledSymbolName() {
  std::string null = typeid(IntegralConstant<U*, nullptr>).name();
  std::string symbol = typeid(IntegralConstant<U*, u>).name();
  return symbol.substr(null.size() - 3, symbol.size() - null.size() + 0);
}


namespace pacxx {
  namespace v2 {

    template<class T, int size, class Enable = void>
    struct vec {
    };

    template<class T, int size>
    struct vec<T, size, typename std::enable_if<(size & (size - 1)) == 0>::type> {
      typedef T type __attribute__((ext_vector_type(size)));
    };

    template<typename F, typename... Args>
    [[reflect]] auto _stage(F func, Args&& ... args) {
      return static_cast<long>(func(std::forward<Args>(args)...));
    }

    template<typename Arg>
    auto stage(const Arg& val) {
      return static_cast<Arg>(_stage([&] { return val; }));
    }

    template<typename T, size_t __sm_size = 0>
    class shared_memory {
    public:
      template<typename U = T,
          typename std::enable_if<!std::is_void<U>::value && __sm_size == 0>::type* = nullptr>
      shared_memory() {
#ifdef __device_code__
        [[shared]] extern T ptr[];
#else
        T* ptr = nullptr;
#endif
        sm_ptr = reinterpret_cast<decltype(sm_ptr)>(ptr);
      }

      template<typename U = T,
          typename std::enable_if<!std::is_void<U>::value && __sm_size != 0>::type* = nullptr>
      shared_memory() {
        [[shared]] T ptr[__sm_size];
        sm_ptr = reinterpret_cast<decltype(sm_ptr)>(ptr);
      }

    private:
      T /*__attribute__((address_space(3)))*/* sm_ptr;

    public:
      T& operator[](int idx) { return sm_ptr[idx]; }

      const T& operator[](int idx) const { return sm_ptr[idx]; }
    };

    template<typename T, typename std::enable_if<!std::is_pointer<
        typename std::remove_reference<T>::type>::value>::type* = nullptr>
    T forward_sec(T val) {
      return val;
    }

    template<typename T, typename std::enable_if<std::is_pointer<
        typename std::remove_reference<T>::type>::value>::type* = nullptr>
    typename std::remove_reference<T>::type forward_sec(T& val) {
      return val;
    }

    // Address space transformations for kernel calls

    template<typename T>
    struct generic_to_global {
      using type = std::conditional_t<
          std::is_reference<T>::value, std::remove_reference_t<T> __attribute((address_space(1))),
          std::conditional_t<
              std::is_pointer<T>::value,
              std::remove_pointer_t<std::remove_reference_t<T>> __attribute((address_space(1)))*, T>>;
    };
    template<typename T>
    struct global_to_generic {
      using type =
      std::conditional_t<std::is_reference<T>::value,
          std::remove_reference_t<T> __attribute((address_space(0))), T>;
    };

    template<typename T, int AS, typename U,
        typename std::enable_if<AS == 1 && std::is_reference<T>::value>::type* = nullptr>
    std::conditional_t<std::is_reference<T>::value,
        std::add_lvalue_reference_t<typename generic_to_global<T>::type>,
        typename generic_to_global<T>::type>
    address_space_cast(U& arg) {
      return *reinterpret_cast<typename generic_to_global<T>::type*>(&arg);
    }

    template<typename T, int AS, typename U,
        typename std::enable_if<AS == 0 && std::is_reference<T>::value>::type* = nullptr>
    T address_space_cast(U& arg) {
      return *reinterpret_cast<std::conditional_t<std::is_const<T>::value, const std::remove_reference_t<T>*,
          std::remove_reference_t<T>* >> (&arg);
    }

    template<typename T, int AS, typename U,
        typename std::enable_if<AS == 1 && std::is_pointer<T>::value>::type* = nullptr>
    auto address_space_cast(U arg) {
      return reinterpret_cast<std::remove_pointer_t<T> __attribute((address_space(1)))*>(arg);
    }

    template<typename T, int AS, typename U,
        typename std::enable_if<AS == 0 && std::is_pointer<T>::value>::type* = nullptr>
    auto address_space_cast(U arg) {
      return reinterpret_cast<T>(arg);
    }

    template<
        typename T, int AS, typename U,
        typename std::enable_if_t<!std::is_reference<T>::value && !std::is_pointer<T>::value, int>* = nullptr>
    T address_space_cast(U val) {
      return static_cast<T>(val);
    }
    //
    // THIS IS THE GENERIC KERNEL EXPRESSION
    //

    template<size_t _C, typename L, typename... ArgTys>
    [[kernel]] void
    genericKernel(L callable, //const __attribute((address_space(1))) L& lambda,
                  std::conditional_t<std::is_reference<ArgTys>::value,
                      std::add_lvalue_reference_t<typename generic_to_global<ArgTys>::type>,
                      typename generic_to_global<ArgTys>::type>... args) noexcept {
       callable(address_space_cast<ArgTys, 0>(args)...);
    }

    template<typename _RuntimeT, size_t _C, typename T>
    struct kernel_caller : public kernel_caller<_RuntimeT, _C, decltype(&T::operator())> {
    };

    template<typename _RuntimeT, size_t _C, typename FType, typename RType, typename... ArgTys>
    struct kernel_caller<_RuntimeT, _C, RType (FType::*)(ArgTys...) const> {
      enum {
        arity = sizeof...(ArgTys)
      };

      template<size_t i>
      struct ArgTy {
        typedef typename std::tuple_element<i, std::tuple<ArgTys...>>::type type;
      };

      using ReturnType = RType;

      template<typename L, typename... Ts>
      static void call(const L& F, const KernelConfiguration& config, Ts&& ... args) {

        // In PACXX V1 the compiler removed the kernel_call in the host code
        // this however, was highly unstable between the two compilation passes and overly complicated.
        // In PACXX V2 we use the __device_code__ macro to distinguish between the two compilation phases
        // This approach has the advantage that we do not have to break up type safety.
#ifdef __device_code__
        genericKernel<_C, L, ArgTys...>(
            address_space_cast<const L &, 1>(F), address_space_cast<ArgTys, 1>(args)...);
#else
        auto& executor = Executor<_RuntimeT>::Create();
        executor.run(typeid(L).name(), config, nullptr, std::forward<Ts>(args)...);
#endif
      }
    };


    template<typename RuntimeT, size_t _C>
    struct exp_kernel_caller {
      template<typename L, typename... Ts>
      static void call(const L& F, const KernelConfiguration& config, Ts&& ... args) {

        // In PACXX V1 the compiler removed the kernel_call in the host code
        // this however, was highly unstable between the two compilation passes and overly complicated.
        // In PACXX V2 we use the __device_code__ macro to distinguish between the two compilation phases
        // This approach has the advantage that we do not have to break up type safety.
#ifdef __device_code__
        genericKernel<_C, L, meta::add_gpu_reference_t<std::remove_reference_t<Ts>>...>(
            /*address_space_cast<const L &, 1>(F)*/ F, address_space_cast<meta::add_gpu_reference_t<std::remove_reference_t<Ts>>, 1>(args)...);
#else
        auto& executor = Executor<RuntimeT>::Create();
        executor.run(F, config, std::forward<Ts>(args)...);
#endif
      }

      template<typename L, typename CallbackTy, typename... Ts>
      static void
      call_with_cb(const L& F, const KernelConfiguration& config, CallbackTy& callback, Ts&& ... args) {

        // In PACXX V1 the compiler removed the kernel_call in the host code
        // this however, was highly unstable between the two compilation passes and overly complicated.
        // In PACXX V2 we use the __device_code__ macro to distinguish between the two compilation phases
        // This approach has the advantage that we do not have to break up type safety.
#ifdef __device_code__
        // call the kernel to force code generation for the template
        genericKernel<_C, L, meta::add_gpu_reference_t<std::remove_reference_t<Ts>>...>(
            /*address_space_cast<const L &, 1>(F)*/ F, address_space_cast<meta::add_gpu_reference_t<std::remove_reference_t<Ts>>, 1>(args)...);
#else
        auto& executor = Executor<RuntimeT>::Create();
        executor.run_with_callback(F, config, callback, std::forward<Ts>(args)...);
#endif
      }
    };

    template<typename L, typename RuntimeT, size_t _C>
    class _kernel {
    public:
      _kernel(const L& lambda, KernelConfiguration config)
          : _function(lambda), _config(config) {
      }


      template<typename... Ts>
      void operator()(Ts&& ... args) const {
        //using caller = kernel_caller<_C, decltype(&std::remove_const_t<std::remove_reference_t<L>>::operator())>;
        using caller = exp_kernel_caller<RuntimeT, _C>;

        caller::call(_function, _config, std::forward<Ts>(args)...);
      }


      auto synchronize() { __message("synchronize not implemented yet"); }

    private:
      const L& _function;
      KernelConfiguration _config;
    };

    template<typename L, typename CB, typename RuntimeT, size_t _C>
    class _kernel_with_cb {
    public:
      _kernel_with_cb(const L& lambda, KernelConfiguration config, CB&& callback)
          : _function(lambda), _config(config), _callback(std::move(callback)) {
      }


      template<typename... Ts>
      void operator()(Ts&& ... args) {
        //using caller = kernel_caller<_C, decltype(&std::remove_const_t<std::remove_reference_t<L>>::operator())>;
        using caller = exp_kernel_caller<RuntimeT, _C>;

        caller::call_with_cb(_function, _config, _callback, std::forward<Ts>(args)...);
      }


      auto synchronize() { __message("synchronize not implemented yet"); }

    private:
      meta::callable_wrapper<L> _function;
      KernelConfiguration _config;
      CB _callback;
    };


    template<typename RuntimeT, typename Func, size_t versioning = __COUNTER__>
    auto kernel(const Func& lambda, KernelConfiguration config) {
      return _kernel<decltype(lambda), RuntimeT, versioning>(lambda, config);
    };

    template<typename Func, typename CallbackFunc, typename RuntimeT, size_t versioning = __COUNTER__>
    auto kernel_with_cb(const Func& lambda, KernelConfiguration config, CallbackFunc&& CB) {
      return _kernel_with_cb<decltype(lambda), CallbackFunc, RuntimeT, versioning>
          (lambda, config, std::forward<CallbackFunc>(CB));
    };

#define __kernel(__lambda, __config, ...)                                                              \
      [](const auto& lambda, pacxx::v2::KernelConfiguration&& config) {    \
          return pacxx::v2::_kernel<decltype(lambda), __COUNTER__>(lambda, std::forward<pacxx::v2::KernelConfiguration>(config));                                  \
      }(__lambda, __config, __VA_ARGS__)

#define __kernel_with_cb(__lambda, __config, __cb, ...)                                                              \
      [](const auto& lambda, pacxx::v2::KernelConfiguration&& config, auto&& CB) {    \
          return pacxx::v2::_kernel_with_cb<decltype(lambda), CallbackFunc, __COUNTER__> \
            (lambda, std::forward<pacxx::v2::KernelConfiguration>(config), std::forward<CallbackFunc>(CB));                                  \
      }(__lambda, __config, __cb, __VA_ARGS__)


  }
}
#endif //PACXX_V2_KERNEL_H_H
