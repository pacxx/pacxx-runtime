//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef PACXX_V2_DEVICEBUFFER_H
#define PACXX_V2_DEVICEBUFFER_H

#include <cstddef>
#include <memory>

namespace pacxx {
namespace v2 {

enum MemAllocMode {
  Standard,
  Unified
};

template <typename T>
class DeviceBufferBase {
public:
  DeviceBufferBase() : src_shadow(nullptr), count_shadow(0), offset_shadow(0) {}

  virtual ~DeviceBufferBase() {}

  virtual T *[[pacxx::device_memory]] get(size_t offset = 0) const = 0;

  virtual void upload(const T *src, size_t count, size_t offset = 0) = 0;
  virtual void download(T *dest, size_t count, size_t offset = 0) = 0;
  virtual void uploadAsync(const T *src, size_t count, size_t offset = 0) = 0;
  virtual void downloadAsync(T *dest, size_t count, size_t offset = 0) = 0;
  virtual void copyTo(T *dest) = 0;
  virtual void enshadow() = 0;
  virtual void restore() = 0;
protected:
  char *src_shadow;
  size_t count_shadow;
  size_t offset_shadow;
};

class RawDeviceBuffer : public DeviceBufferBase<void> {};

template <typename T> class DeviceBuffer : public DeviceBufferBase<T> {
public:
  DeviceBuffer(std::unique_ptr<RawDeviceBuffer> buffer) : _buffer(std::move(buffer)) {}

  RawDeviceBuffer *getRawBuffer() { return _buffer.get(); }

  virtual ~DeviceBuffer() {}

  DeviceBuffer(const DeviceBuffer &) = delete;

  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  DeviceBuffer(DeviceBuffer &&rhs) { _buffer = std::move(rhs._buffer); }

  DeviceBuffer &operator=(DeviceBuffer &&rhs) {
    _buffer = std::move(rhs._buffer);
    return *this;
  }

  virtual T* [[pacxx::device_memory]] get(size_t offset = 0) const final {
    return reinterpret_cast<T *>(_buffer->get(sizeof(T) * offset));
  }

  virtual void upload(const T *src, size_t count, size_t offset = 0) override {
    _buffer->upload(src, count * sizeof(T), offset);
  }

  virtual void download(T *dest, size_t count, size_t offset = 0) override {
    _buffer->download(dest, count * sizeof(T), offset);
  }

  virtual void uploadAsync(const T *src, size_t count,
                           size_t offset = 0) override {
    _buffer->uploadAsync(src, count * sizeof(T), offset);
  }

  virtual void downloadAsync(T *dest, size_t count,
                             size_t offset = 0) override {
    _buffer->downloadAsync(dest, count * sizeof(T), offset);
  }

  virtual void copyTo(T *dest) override { _buffer->copyTo(dest); }

  virtual void enshadow() override { _buffer->enshadow(); }

  virtual void restore() override { _buffer->restore(); }

private:
  std::unique_ptr<RawDeviceBuffer> _buffer;
};
}
}

#endif // PACXX_V2_DEVICEBUFFER_H
