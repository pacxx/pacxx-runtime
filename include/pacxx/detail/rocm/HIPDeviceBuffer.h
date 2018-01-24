//
// Created by mhaidl on 02/06/16.
//

#include "pacxx/detail/DeviceBuffer.h"
#include "pacxx/detail/common/Log.h"
#include <memory>

#ifndef PACXX_V2_HIPDEVICEBUFFER_H
#define PACXX_V2_HIPDEVICEBUFFER_H
namespace pacxx {
namespace v2 {
class HIPRawDeviceBuffer : public RawDeviceBuffer {
public:
  HIPRawDeviceBuffer(size_t size);

  virtual ~HIPRawDeviceBuffer();

  HIPRawDeviceBuffer(const HIPRawDeviceBuffer &) = delete;

  HIPRawDeviceBuffer &operator=(const HIPRawDeviceBuffer &) = delete;

  HIPRawDeviceBuffer(HIPRawDeviceBuffer &&rhs);

  HIPRawDeviceBuffer &operator=(HIPRawDeviceBuffer &&rhs);

  virtual void *get(size_t offset = 0) const final;

  virtual void upload(const void *src, size_t bytes,
                      size_t offset = 0) override;

  virtual void download(void *dest, size_t bytes, size_t offset = 0) override;

  virtual void uploadAsync(const void *src, size_t bytes,
                           size_t offset = 0) override;

  virtual void downloadAsync(void *dest, size_t bytes,
                             size_t offset = 0) override;

  virtual void copyTo(void *dest) override;

private:
  [[pacxx::device_memory]] char *_buffer;
  size_t _size;
};
}
}
#endif // PACXX_V2_HIPDEVICEBUFFER_H
