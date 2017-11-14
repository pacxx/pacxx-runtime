//===-----------------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once
#include <cstddef>

namespace pacxx {
namespace v2 {

template<class T, int size, class Enable = void> struct vec {};

template<class T, int size>
struct vec<T, size, typename std::enable_if<(size & (size - 1)) == 0>::type> {
  typedef T type __attribute__((ext_vector_type(size)));
};

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

}
}