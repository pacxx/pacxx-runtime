//
// Created by mhaidl on 02/06/16.
//

#include "CUDAErrorDetection.h"
#include <detail/DeviceBuffer.h>
#include "detail/common/Log.h"
#include <memory>

#ifndef PACXX_V2_CUDADEVICEBUFFER_H
#define PACXX_V2_CUDADEVICEBUFFER_H
namespace pacxx {
namespace v2 {
class CUDARawDeviceBuffer {
  friend class CUDARuntime;

private:
  CUDARawDeviceBuffer() {
  }

  void allocate (size_t bytes)
  {
    SEC_CUDA_CALL(cudaMalloc(&_buffer, bytes));
  }

public:
  virtual ~CUDARawDeviceBuffer() {
    if (_buffer)
      SEC_CUDA_CALL(cudaFree(_buffer));
    else
      __message("dtor but null");
  }

  CUDARawDeviceBuffer(const CUDARawDeviceBuffer &) = delete;
  CUDARawDeviceBuffer &operator=(const CUDARawDeviceBuffer &) = delete;

  CUDARawDeviceBuffer(CUDARawDeviceBuffer &&rhs) {
    _buffer = rhs._buffer;
    rhs._buffer = nullptr;
  }
  CUDARawDeviceBuffer &operator=(CUDARawDeviceBuffer &&rhs) {
    _buffer = rhs._buffer;
    rhs._buffer = nullptr;
    return *this;
  }

  void *get() { return _buffer; }

  void upload(void *src, size_t bytes) {
    SEC_CUDA_CALL(cudaMemcpy(_buffer, src, bytes, cudaMemcpyHostToDevice));
  }
  void download(void *dest, size_t bytes) {
    SEC_CUDA_CALL(cudaMemcpy(dest, _buffer, bytes, cudaMemcpyDeviceToHost));
  }
  void uploadAsync(void *src, size_t bytes) {
    SEC_CUDA_CALL(cudaMemcpyAsync(_buffer, src, bytes, cudaMemcpyHostToDevice));
  }
  void downloadAsync(void *dest, size_t bytes) {
    SEC_CUDA_CALL(
        cudaMemcpyAsync(dest, _buffer, bytes, cudaMemcpyDeviceToHost));
  }

private:
  void *_buffer;
};

template <typename T> class CUDADeviceBuffer : public DeviceBuffer<T> {
  friend class CUDARuntime;

private:
  CUDADeviceBuffer(CUDARawDeviceBuffer buffer) : _buffer(std::move(buffer)) {}

public:
  virtual ~CUDADeviceBuffer() {}

  CUDADeviceBuffer(const CUDADeviceBuffer &) = delete;
  CUDADeviceBuffer &operator=(const CUDADeviceBuffer &) = delete;

  CUDADeviceBuffer(CUDADeviceBuffer &&rhs) { _buffer = std::move(rhs._buffer); }
  CUDADeviceBuffer &operator=(CUDADeviceBuffer &&rhs) {
    _buffer = std::move(rhs._buffer);
    return *this;
  }

  virtual T *get() final { return reinterpret_cast<T *>(_buffer.get()); }

  virtual void upload(T *src, size_t count) override {
    _buffer.upload(src, count * sizeof(T));
  }
  virtual void download(T *dest, size_t count) override {
    _buffer.download(dest, count * sizeof(T));
  }
  virtual void uploadAsync(T *src, size_t count) override {
    _buffer.uploadAsync(src, count * sizeof(T));
  }
  virtual void downloadAsync(T *dest, size_t count) override {
    _buffer.downloadAsync(dest, count * sizeof(T));
  }

private:
  CUDARawDeviceBuffer _buffer;
};
}
}
#endif // PACXX_V2_CUDADEVICEBUFFER_H
