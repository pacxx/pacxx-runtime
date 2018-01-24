//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef PACXX_V2_NATIVEDEVICEBUFFER_H
#define PACXX_V2_NATIVEDEVICEBUFFER_H

#include "pacxx/detail/DeviceBuffer.h"
#include "pacxx/detail/common/Log.h"
#include <memory>
#include <functional>

namespace pacxx {
namespace v2 {

class NativeRawDeviceBuffer : public RawDeviceBuffer {
public:
  NativeRawDeviceBuffer(size_t size, unsigned padding);

  virtual ~NativeRawDeviceBuffer();

  NativeRawDeviceBuffer(const NativeRawDeviceBuffer &) = delete;

  NativeRawDeviceBuffer &operator=(const NativeRawDeviceBuffer &) = delete;

  NativeRawDeviceBuffer(NativeRawDeviceBuffer &&rhs);

  NativeRawDeviceBuffer &operator=(NativeRawDeviceBuffer &&rhs);

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
  char *[[pacxx::device_memory]] _buffer;
  size_t _size;
};
} // v2 namespace
} // pacxx namespace

#endif // PACXX_V2_NATIVEDEVICEBUFFER_H
