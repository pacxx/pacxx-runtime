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
  RemoteRawDeviceBuffer(std::function<void(RemoteRawDeviceBuffer&)> deleter, RemoteRuntime* runtime, MemAllocMode mode = Standard);

  void allocate(size_t bytes);

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

  virtual void restore() override;

  virtual void abandon() override;

  virtual void mercy() override;

private:
  [[pacxx::device_memory]] char *_buffer;
  size_t _size;
  unsigned _mercy;
  MemAllocMode _mode;
  std::function<void(RemoteRawDeviceBuffer&)> _deleter;
  RemoteRuntime* _runtime;
};
}
}
