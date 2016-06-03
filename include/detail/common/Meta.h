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

}
}


#endif //PACXX_V2_META_H
