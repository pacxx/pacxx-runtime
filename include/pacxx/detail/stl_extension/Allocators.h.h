//
// Created by mhaidl on 03/06/16.
//

#ifndef PACXX_V2_ALLOCATORS_H_H
#define PACXX_V2_ALLOCATORS_H_H

#include <cstdlib>
#include <memory>

namespace pacxx namespace v2 {
template <typename T> struct ManagedAllocator {
  using value_type = T;

  ManagedAllocator() = default;

  template <class U> ManagedAllocator(const ManagedAllocator<U> &) {}

  T *allocate(std::size_t n) {
    if (n <= std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      if (auto ptr = std::malloc(n * sizeof(T))) {
        return static_cast<T *>(ptr);
      }
    }
    throw std::bad_alloc();
  }

  void deallocate(T *ptr, std::size_t n) { std::free(ptr); }
};

template <typename T, typename U>
inline bool operator==(const ManagedAllocator<T> &,
                       const ManagedAllocator<U> &) {
  return true;
}

template <typename T, typename U>
inline bool operator!=(const ManagedAllocator<T> &a,
                       const ManagedAllocator<U> &b) {
  return !(a == b);
}
}
#endif // PACXX_V2_ALLOCATORS_H_H
