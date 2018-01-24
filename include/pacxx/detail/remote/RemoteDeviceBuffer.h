//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "pacxx/detail/DeviceBuffer.h"
#include "pacxx/detail/common/Log.h"
#include <memory>

#pragma once

namespace pacxx {
namespace v2 {

class RemoteRuntime;

class RemoteRawDeviceBuffer : public RawDeviceBuffer {
public:
  RemoteRawDeviceBuffer(size_t size, RemoteRuntime* runtime);

  virtual ~RemoteRawDeviceBuffer();

  RemoteRawDeviceBuffer(const RemoteRawDeviceBuffer &) = delete;

  RemoteRawDeviceBuffer &operator=(const RemoteRawDeviceBuffer &) = delete;

  RemoteRawDeviceBuffer(RemoteRawDeviceBuffer &&rhs);

  RemoteRawDeviceBuffer &operator=(RemoteRawDeviceBuffer &&rhs);

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
  RemoteRuntime* _runtime;
};
}
}

