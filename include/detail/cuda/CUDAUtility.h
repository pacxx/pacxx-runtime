//
// Created by mhaidl on 14/06/16.
//

#ifndef PACXX_V2_UTILITY_H
#define PACXX_V2_UTILITY_H

#include <cstddef>
#include <cuda.h>
#include "CUDAErrorDetection.h"

namespace pacxx
{
namespace cuda
{
    template <class T>
    struct CUDAPinnedAllocator {
      typedef T value_type;
      CUDAPinnedAllocator(){}
      template <class U> CUDAPinnedAllocator(const CUDAPinnedAllocator<U>& other){}

      T* allocate(std::size_t n)
      {
        T* ptr = nullptr;

        SEC_CUDA_CALL(cudaHostAlloc(&ptr, n * sizeof(T), cudaHostAllocDefault));

        return ptr;

      }
      void deallocate(T* ptr, std::size_t n){
        SEC_CUDA_CALL(cudaFreeHost(ptr));
      }
    };
    template <class T, class U>
    bool operator==(const CUDAPinnedAllocator<T>&, const CUDAPinnedAllocator<U>&) { return true; };

    template <class T, class U>
    bool operator!=(const CUDAPinnedAllocator<T>&, const CUDAPinnedAllocator<U>&){ return false; };

  template <class T>
  struct CUDAManagedAllocator {
    typedef T value_type;
    CUDAManagedAllocator(){}
    template <class U> CUDAManagedAllocator(const CUDAManagedAllocator<U>& other){}

    T* allocate(std::size_t n)
    {
      T* ptr = nullptr;

      SEC_CUDA_CALL(cudaMallocManaged(&ptr, n * sizeof(T) + 256, cudaMemAttachGlobal));

      ptr = static_cast<T*>(static_cast<char*>(ptr) + 256);
      return ptr;

    }
    void deallocate(T* ptr, std::size_t n){
      SEC_CUDA_CALL(cudaFree(static_cast<char*>(ptr)-256));
    }
  };
  template <class T, class U>
  bool operator==(const CUDAManagedAllocator<T>&, const CUDAManagedAllocator<U>&) { return true; };

  template <class T, class U>
  bool operator!=(const CUDAManagedAllocator<T>&, const CUDAManagedAllocator<U>&){ return false; };
}
}


#endif //PACXX_V2_UTILITY_H
