//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once
//#include <device_types.h>
#ifndef __forceinline__
#define __forceinline__ __attribute__((always_inline))
#endif

#include <tuple>
#include <type_traits>

//////////////////////////////// PRINTF ////////////////////////////////

extern "C" void _printf(const char *, void *);

template <typename T> struct extend_type {

  using type = typename std::conditional<
      std::is_integral<T>::value,
      typename std::conditional<std::is_same<T, short>::value, int, T>::type,
      typename std::conditional<std::is_same<T, float>::value, double,
                                T>::type>::type;
};

namespace pacxx {
namespace nvidia {
template <typename... Args> void printf(const char *str, Args... args) {
#ifdef __device_code__
  std::tuple<typename extend_type<Args>::type..., int> tpl(args..., 0);
  _printf(str, reinterpret_cast<void *>(&tpl));
#endif
}
} // namespace nvidia
namespace native {
template <typename... Args> void printf(const char *str, Args... args) {
#ifdef __device_code__
  printf(str, args...);
#endif
}
} // namespace nvidia
} // namespace native
