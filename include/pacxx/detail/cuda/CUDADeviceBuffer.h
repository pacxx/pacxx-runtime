//===-----------------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CUDAErrorDetection.h"
#include "pacxx/detail/DeviceBuffer.h"
#include "pacxx/detail/common/Log.h"
#include <cuda_runtime_api.h>
#include <memory>

#ifndef PACXX_V2_CUDADEVICEBUFFER_H
#define PACXX_V2_CUDADEVICEBUFFER_H
namespace pacxx {
namespace v2 {
class CUDARawDeviceBuffer : public RawDeviceBuffer {
  friend class CUDARuntime;

private:
  CUDARawDeviceBuffer(std::function<void(CUDARawDeviceBuffer&)> deleter, MemAllocMode mode = Standard);

  void allocate(size_t bytes);

public:
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

  virtual void abandon() override;

  virtual void mercy() override;

private:
  [[pacxx::device_memory]] char *_buffer;
  size_t _size;
  unsigned _mercy;
  MemAllocMode _mode;
  std::function<void(CUDARawDeviceBuffer&)> _deleter;
};

template <typename T> class CUDADeviceBuffer : public DeviceBuffer<T> {
  friend class CUDARuntime;

private:
  CUDADeviceBuffer(CUDARawDeviceBuffer buffer) : _buffer(std::move(buffer)) {}

  CUDARawDeviceBuffer *getRawBuffer() { return &_buffer; }

public:
  virtual ~CUDADeviceBuffer() {}

  CUDADeviceBuffer(const CUDADeviceBuffer &) = delete;

  CUDADeviceBuffer &operator=(const CUDADeviceBuffer &) = delete;

  CUDADeviceBuffer(CUDADeviceBuffer &&rhs) { _buffer = std::move(rhs._buffer); }

  CUDADeviceBuffer &operator=(CUDADeviceBuffer &&rhs) {
    _buffer = std::move(rhs._buffer);
    return *this;
  }

  virtual T* [[pacxx::device_memory]] get(size_t offset = 0) const final {
    return reinterpret_cast<T *>(_buffer.get(sizeof(T) * offset));
  }

  virtual void upload(const T *src, size_t count, size_t offset = 0) override {
    _buffer.upload(src, count * sizeof(T), offset);
  }

  virtual void download(T *dest, size_t count, size_t offset = 0) override {
    _buffer.download(dest, count * sizeof(T), offset);
  }

  virtual void uploadAsync(const T *src, size_t count,
                           size_t offset = 0) override {
    _buffer.uploadAsync(src, count * sizeof(T), offset);
  }

  virtual void downloadAsync(T *dest, size_t count,
                             size_t offset = 0) override {
    _buffer.downloadAsync(dest, count * sizeof(T), offset);
  }

  virtual void abandon() override { _buffer.abandon(); }

  virtual void mercy() override { _buffer.mercy(); }

  virtual void copyTo(T *dest) override { _buffer.copyTo(dest); }

private:
  CUDARawDeviceBuffer _buffer;
};
}
}
#endif // PACXX_V2_CUDADEVICEBUFFER_H
