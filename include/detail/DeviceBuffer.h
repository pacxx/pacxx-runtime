//
// Created by mhaidl on 01/06/16.
//

#ifndef PACXX_V2_DEVICEBUFFER_H
#define PACXX_V2_DEVICEBUFFER_H

#include <cstddef>

namespace pacxx
{
namespace v2
{
  class DeviceBufferBase{
  public:
    virtual ~DeviceBufferBase() {}
  };

  template <typename T>
  class DeviceBuffer : public DeviceBufferBase
  {
  public:
    virtual ~DeviceBuffer(){}

    virtual T* get() = 0;

    virtual void upload(T* src, size_t count) = 0;
    virtual void download(T* dest, size_t count) = 0;
    virtual void uploadAsync(T* src, size_t count) = 0;
    virtual void downloadAsync(T* dest, size_t count) = 0;
  };
}
}


#endif //PACXX_V2_DEVICEBUFFER_H
