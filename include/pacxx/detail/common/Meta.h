//
// Created by mhaidl on 02/06/16.
//

#ifndef PACXX_V2_META_H
#define PACXX_V2_META_H

#include "../DeviceBuffer.h"
#include <type_traits>
#include <vector>

namespace pacxx {
namespace meta {

template <typename T> struct callable_wrapper {
  T callable;

  const T &get() const { return callable; }

  callable_wrapper(T callable) : callable(callable) {}

  template<typename... Ts> auto operator()(Ts &&... args) const {
    return callable(std::forward<Ts>(args)...);
  }
};

template <class T> struct is_vector : std::false_type {};

template <class T, class Alloc>
struct is_vector<std::vector<T, Alloc>> : std::true_type {};

template <class T> struct is_wrapped_callable : std::false_type {};

template <class T>
struct is_wrapped_callable<pacxx::meta::callable_wrapper<T>> : std::true_type {
};

template <typename T>
struct is_devbuffer : std::is_base_of<v2::DeviceBufferBase, T> {};

// since genericKernel takes all parameters as lvalue and not as xvalue we let
// the types of form int*&& decay to int* to avoid the automatic decay to int**
template <typename T> struct add_gpu_reference {
  using type = std::conditional_t<is_vector<std::decay_t<T>>::value,
                                  std::add_lvalue_reference_t<T>, T>;
};

template <typename T>
using add_gpu_reference_t = typename add_gpu_reference<T>::type;
}
}

#endif // PACXX_V2_META_H
