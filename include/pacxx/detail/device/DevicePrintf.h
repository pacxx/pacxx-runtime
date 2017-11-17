//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

//#include <tuple>
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
namespace meta {
template <typename T, typename TT = typename std::remove_reference<T>::type,
          size_t... I>
auto reverse_impl(T &&t, std::index_sequence<I...>) -> std::tuple<
    typename std::tuple_element<sizeof...(I) - 1 - I, TT>::type...> {
  return std::make_tuple(std::get<sizeof...(I) - 1 - I>(std::forward<T>(t))...);
}

template <typename T, typename TT = typename std::remove_reference<T>::type>
auto reverse(T &&t) -> decltype(
    reverse_impl(std::forward<T>(t),
                 std::make_index_sequence<std::tuple_size<TT>::value>())) {
  return reverse_impl(std::forward<T>(t),
                      std::make_index_sequence<std::tuple_size<TT>::value>());
}
} // namespace meta
namespace nvidia {
template <typename... Args> void printf(const char *str, Args... args) {
#ifdef __device_code__
  std::tuple<typename extend_type<Args>::type..., int> tpl(args..., 0);
  _printf(str, reinterpret_cast<void *>(&meta::reverse(tpl)));
#endif
}
} // namespace nvidia
namespace native {
template <typename... Args> void printf(const char *str, Args &&... args) {
#ifdef __device_code__
  ::printf(str, std::forward<Args>(args)...);
#endif
}
} // namespace native

template <typename... Args> void printf(const char *str, Args &&... args) {
  switch (__pacxx_backend_id()) {
  case 0:
    pacxx::nvidia::printf(str, std::forward<Args>(args)...);
    break;
  case 1:
    pacxx::native::printf(str, args...);
    break;
  case 2: // AMD not implemented yet
    break;
  default:
    break;
  }
}

} // namespace pacxx
