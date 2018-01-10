//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef PACXX_V2_META_H
#define PACXX_V2_META_H

#include "pacxx/detail/DeviceBuffer.h"
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

}
}

#endif // PACXX_V2_META_H
