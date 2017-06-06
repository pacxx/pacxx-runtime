//
// Created by mhaidl on 01/06/16.
//

#ifndef PACXX_V2_DEVICEBUFFER_H
#define PACXX_V2_DEVICEBUFFER_H

#include <cstddef>

namespace pacxx {
namespace v2 {

enum MemAllocMode {
  Standard,
  Unified
};


class DeviceBufferBase {
public:
  virtual ~DeviceBufferBase() {}
};

template <typename T> class DeviceBuffer : public DeviceBufferBase {
public:
  virtual ~DeviceBuffer() {}

  virtual T *get(size_t offset = 0) const = 0;

  virtual void upload(const T *src, size_t count, size_t offset = 0) = 0;
  virtual void download(T *dest, size_t count, size_t offset = 0) = 0;
  virtual void uploadAsync(const T *src, size_t count, size_t offset = 0) = 0;
  virtual void downloadAsync(T *dest, size_t count, size_t offset = 0) = 0;
  virtual void copyTo(T *dest) = 0;

  virtual void mercy() = 0;
  virtual void abandon() = 0;
};

class RawDeviceBuffer : public DeviceBuffer<void> {};
}
}

#endif // PACXX_V2_DEVICEBUFFER_H
