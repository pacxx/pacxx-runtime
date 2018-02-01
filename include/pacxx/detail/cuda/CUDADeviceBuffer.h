//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef PACXX_V2_CUDADEVICEBUFFER_H
#define PACXX_V2_CUDADEVICEBUFFER_H

#include "pacxx/detail/DeviceBuffer.h"
#include "pacxx/detail/common/Log.h"
#include <memory>

namespace pacxx {
namespace v2 {

class CUDARuntime;

class CUDARawDeviceBuffer : public RawDeviceBuffer {
public:
  CUDARawDeviceBuffer(size_t size, CUDARuntime* runtime, MemAllocMode mode = Standard);

  virtual ~CUDARawDeviceBuffer();

  CUDARawDeviceBuffer(const CUDARawDeviceBuffer &) = delete;

  CUDARawDeviceBuffer &operator=(const CUDARawDeviceBuffer &) = delete;

  CUDARawDeviceBuffer(CUDARawDeviceBuffer &&rhs);

  CUDARawDeviceBuffer &operator=(CUDARawDeviceBuffer &&rhs);

  virtual void *get(size_t offset = 0) const final;

  virtual void upload(const void *src, size_t bytes,
                      size_t offset = 0) override;

  virtual void download(void *dest, size_t bytes, size_t offset = 0) override;

  virtual void uploadAsync(const void *src, size_t bytes,
                           size_t offset = 0) override;

  virtual void downloadAsync(void *dest, size_t bytes,
                             size_t offset = 0) override;

  virtual void copyTo(void *dest) override;

  virtual void enshadow() override;

  virtual void restore() override;

private:
  [[pacxx::device_memory]] char *_buffer;
  size_t _size;
  CUDARuntime* _runtime;
  MemAllocMode _mode;
};
}
}

#endif // PACXX_V2_CUDADEVICEBUFFER_H
