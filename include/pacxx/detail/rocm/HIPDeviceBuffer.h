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
  friend class HIPRuntime;

private:
  HIPRawDeviceBuffer(std::function<void(HIPRawDeviceBuffer&)> deleter, MemAllocMode mode = Standard);

  void allocate(size_t bytes);

public:
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

  virtual void abandon() override;

  virtual void mercy() override;

private:
  [[pacxx::device_memory]] char *_buffer;
  size_t _size;
  unsigned _mercy;
  MemAllocMode _mode;
  std::function<void(HIPRawDeviceBuffer&)> _deleter;
};

template <typename T> class HIPDeviceBuffer : public DeviceBuffer<T> {
  friend class HIPRuntime;

private:
  HIPDeviceBuffer(HIPRawDeviceBuffer buffer) : _buffer(std::move(buffer)) {}

  HIPRawDeviceBuffer *getRawBuffer() { return &_buffer; }

public:
  virtual ~HIPDeviceBuffer() {}

  HIPDeviceBuffer(const HIPDeviceBuffer &) = delete;

  HIPDeviceBuffer &operator=(const HIPDeviceBuffer &) = delete;

  HIPDeviceBuffer(HIPDeviceBuffer &&rhs) { _buffer = std::move(rhs._buffer); }

  HIPDeviceBuffer &operator=(HIPDeviceBuffer &&rhs) {
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
  HIPRawDeviceBuffer _buffer;
};
}
}
#endif // PACXX_V2_HIPDEVICEBUFFER_H
