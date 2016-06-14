//
// Created by mhaidl on 02/06/16.
//

#ifndef PACXX_V2_MEMORYMANAGER_H
#define PACXX_V2_MEMORYMANAGER_H

#include <memory>
#include <map>
#include <detail/common/Log.h>
#include "IRRuntime.h"
#include "common/Meta.h"

namespace pacxx
{
namespace v2
{
  class MemoryManager
  {
  public:
    MemoryManager(IRRuntimeBase& runtime) : _runtime(runtime) {}

    virtual ~MemoryManager(){}

    template <typename T, typename Allocator>
    RawDeviceBuffer& manageVector(const std::vector<T, Allocator>& vec)
    {
      auto buffer = _managed_vectors[reinterpret_cast<const void*>(&vec)];
      if (!buffer) {
        // allocate memory for the vector and the data of the vector
        size_t additonal_bytes = std::max(sizeof(std::vector<T>), _runtime.getPreferedMemoryAlignment());
        size_t vector_memsize = vec.size() * sizeof(T);
        buffer = _runtime.allocateRawMemory(additonal_bytes + vector_memsize);

        struct { char* begin, end; } fake_vector;
        fake_vector.begin = reinterpret_cast<char *>(buffer->get(additonal_bytes));
        fake_vector.end = fake_vector.end + vector_memsize;

        buffer->uploadAsync(&fake_vector, sizeof(fake_vector), additonal_bytes - 32); // TODO: fix the 32 here
        buffer->upload(vec.data(), vector_memsize, additonal_bytes);

        _managed_vectors[reinterpret_cast<const void*>(&vec)] = buffer;
      }
      return *buffer;
    }


    template <typename T, typename Allocator>
    RawDeviceBuffer& translateVector(const std::vector<T, Allocator>& vec)
    {
      auto ptr = _managed_vectors[reinterpret_cast<const void*>(&vec)];
      if (!ptr)
        throw common::generic_exception("unmanaged vector instance provided to translateVector");
      return *ptr;
    }

    // make a vector unmanaged all memory on the device is been
    // freed and the vector is removed from the managed vectors
    template <typename T, typename Allocator>
    void unmanageVector(const std::vector<T, Allocator>& vec)
    {
      auto ptr = _managed_vectors[reinterpret_cast<const void*>(&vec)];
      if (ptr) {
        _runtime.deleteRawMemory(ptr);
        _managed_vectors[reinterpret_cast<const void*>(&vec)] = nullptr;

      }
      else __error("unmanaged vector supplied");
    }

    template<typename L>
    RawDeviceBuffer& getTemporaryLambda(const L& lambda){
      auto buffer = _temporaries[reinterpret_cast<const void*>(&lambda)];
      if (!buffer) {
        buffer = _runtime.allocateRawMemory(sizeof(L));
        buffer->uploadAsync(&lambda, sizeof(L));
        _temporaries[reinterpret_cast<const void*>(&lambda)] = buffer;
      }
      return *buffer;
    }

  private:
    IRRuntimeBase& _runtime;
    std::map<const void*, RawDeviceBuffer*> _managed_vectors;
    std::map<const void*, RawDeviceBuffer*> _temporaries;
  };
}

  namespace meta
  {
    struct memory_translation
    {
      template <typename T, std::enable_if_t<is_vector<T>::value>* = nullptr>
      auto operator() (v2::MemoryManager& mm, const T& data)
      {
          return mm.manageVector(data).get(256); // TODO: make the 256 dynamic
      }

      template <typename T, std::enable_if_t<is_devbuffer<T>::value>* = nullptr>
      auto operator() (v2::MemoryManager& mm, const T& data)
      {
        return data.get();
      }

      template <typename T, std::enable_if_t<!(is_vector<T>::value || is_devbuffer<T>::value)>* = nullptr>
      auto operator() (v2::MemoryManager& mm, const T& data)
      {
        return data;
      }
    };
  }
}
#endif //PACXX_V2_MEMORYMANAGER_H
