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

    template <typename T>
    RawDeviceBuffer& manageVector(const std::vector<T>& vec)
    {
      auto buffer = _managed_vectors[reinterpret_cast<const void*>(&vec)];
      if (!buffer) {
        // allocate memory for the vector and the data of the vector
        size_t header_offset = sizeof(std::vector<T>);
        size_t additonal_bytes = std::max(header_offset, _runtime.getPreferedMemoryAlignment());
        size_t vector_memsize = vec.size() * sizeof(T);

        buffer = _runtime.allocateRawMemory(additonal_bytes + vector_memsize);
        char *this_ptr = reinterpret_cast<char *>(buffer->get(additonal_bytes));
        buffer->upload(vec.data(), vector_memsize, additonal_bytes);
        buffer->upload(&this_ptr, sizeof(intptr_t), additonal_bytes - 32); // TODO: fix the 32 here
        this_ptr += vector_memsize;
        buffer->upload(&this_ptr, sizeof(intptr_t), additonal_bytes - 24); // TODO: fix the 24 here
        _managed_vectors[reinterpret_cast<const void*>(&vec)] = buffer;
      }
      return *buffer;
    }

    // make a vector unmanaged all memory on the device is been
    // freed and the vector is removed from the managed vectors
    template <typename T>
    void unmanageVector(const std::vector<T>& vec)
    {
      auto ptr = _managed_vectors[reinterpret_cast<const void*>(&vec)];
      if (ptr)
        _runtime.deleteRawMemory(ptr);
      else __error("unmanaged vector supplied");
    }

  private:
    IRRuntimeBase& _runtime;
    std::map<const void*, RawDeviceBuffer*> _managed_vectors;
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
