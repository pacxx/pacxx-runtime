//
// Created by mhaidl on 02/06/16.
//

#ifndef PACXX_V2_META_H
#define PACXX_V2_META_H

#include <type_traits>
#include <vector>
#include "../DeviceBuffer.h"


namespace pacxx
{
namespace meta
{
  template <class T> struct is_vector : std::false_type {
  };

  template <class T, class Alloc> struct is_vector<std::vector<T, Alloc>> : std::true_type {
};

  template <typename T> struct is_devbuffer : std::is_base_of<v2::DeviceBufferBase, T> {};

  // since genericKernel takes all parameters as lvalue and not as xvalue we let
  // the types of form int*&& decay to int* to avoid the automatic decay to int**
  template <typename T>
  struct remove_reference
  {
    using type = std::conditional_t<std::is_pointer<std::decay_t<T>>::value || std::is_arithmetic<std::decay_t<T>>::value, std::decay_t<T>, T>;
  };

  template<typename T>
  using remove_reference_t = typename remove_reference<T>::type;


}
}


#endif //PACXX_V2_META_H
