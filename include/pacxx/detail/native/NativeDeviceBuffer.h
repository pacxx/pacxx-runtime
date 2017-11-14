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

  friend class NativeRuntime;

private:
  NativeRawDeviceBuffer(std::function<void(NativeRawDeviceBuffer&)> deleter);

  void allocate(size_t bytes, unsigned padding = 0);

  void allocate(size_t bytes, char *host_ptr);

public:
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

  virtual void abandon() override;

  virtual void mercy() override;

private:
  char *[[pacxx::device_memory]] _buffer;
  size_t _size;
  unsigned _mercy;
  bool _isHost;
  std::function<void(NativeRawDeviceBuffer&)> _deleter;
};

template <typename T> class NativeDeviceBuffer : public DeviceBuffer<T> {
  friend class NativeRuntime;

private:
  NativeDeviceBuffer(NativeRawDeviceBuffer buffer)
      : _buffer(std::move(buffer)) {}

  NativeRawDeviceBuffer *getRawBuffer() { return &_buffer; }

public:
  virtual ~NativeDeviceBuffer() {}

  NativeDeviceBuffer(const NativeDeviceBuffer &) = delete;

  NativeDeviceBuffer &operator=(const NativeDeviceBuffer &) = delete;

  NativeDeviceBuffer(NativeDeviceBuffer &&rhs) {
    _buffer = std::move(rhs._buffer);
  }

  NativeDeviceBuffer &operator=(NativeDeviceBuffer &&rhs) {
    _buffer = std::move(rhs._buffer);
    return *this;
  }

  virtual T *[[pacxx::device_memory]] get(size_t offset = 0) const final {
    return reinterpret_cast<T *>(_buffer.get(sizeof(T) * offset));
  }

  virtual void upload(const T *src, size_t count, size_t offset = 0) override {
    _buffer.upload(src, count * sizeof(T), offset);
  }

  virtual void download(T *dest, size_t count, size_t offset = 0) override {
    __message("downloading ", count * sizeof(T));
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
  NativeRawDeviceBuffer _buffer;
};
} // v2 namespace
} // pacxx namespace

#endif // PACXX_V2_NATIVEDEVICEBUFFER_H
