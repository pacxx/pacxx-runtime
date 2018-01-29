//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef PACXX_V2_HIPDEVICEBUFFER_H
#define PACXX_V2_HIPDEVICEBUFFER_H

#include "pacxx/detail/DeviceBuffer.h"
#include "pacxx/detail/common/Log.h"
#include <memory>

namespace pacxx {
namespace v2 {

class HIPRuntime;

class HIPRawDeviceBuffer : public RawDeviceBuffer {
public:
  HIPRawDeviceBuffer(size_t size, HIPRuntime* runtime);

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

  virtual void restore() override;

private:
  [[pacxx::device_memory]] char *_buffer;
  size_t _size;
  HIPRuntime* _runtime;
};
}
}

#endif // PACXX_V2_HIPDEVICEBUFFER_H
