//
// Created by mhaidl on 02/06/16.
//

#include "CUDAErrorDetection.h"
#include <detail/DeviceBuffer.h>
#include "detail/common/Log.h"
#include <cuda_runtime_api.h>
#include <memory>

#ifndef PACXX_V2_CUDADEVICEBUFFER_H
#define PACXX_V2_CUDADEVICEBUFFER_H
namespace pacxx {
  namespace v2 {
    class CUDARawDeviceBuffer : public RawDeviceBuffer {
      friend class CUDARuntime;

    private:
      CUDARawDeviceBuffer();

      void allocate(size_t bytes);

    public:
      virtual ~CUDARawDeviceBuffer();

      CUDARawDeviceBuffer(const CUDARawDeviceBuffer&) = delete;

      CUDARawDeviceBuffer& operator=(const CUDARawDeviceBuffer&) = delete;

      CUDARawDeviceBuffer(CUDARawDeviceBuffer&& rhs);

      CUDARawDeviceBuffer& operator=(CUDARawDeviceBuffer&& rhs);

      virtual void* get(size_t offset = 0) const final;

      virtual void upload(const void* src, size_t bytes, size_t offset = 0) override;

      virtual void download(void* dest, size_t bytes, size_t offset = 0) override;

      virtual void uploadAsync(const void* src, size_t bytes, size_t offset = 0) override;

      virtual void downloadAsync(void* dest, size_t bytes, size_t offset = 0) override;

      virtual void abandon() override;

    private:
      char* _buffer;
    };

    template<typename T>
    class CUDADeviceBuffer : public DeviceBuffer<T> {
      friend class CUDARuntime;

    private:
      CUDADeviceBuffer(CUDARawDeviceBuffer buffer) : _buffer(std::move(buffer)) {}

      CUDARawDeviceBuffer* getRawBuffer() { return &_buffer; }

    public:
      virtual ~CUDADeviceBuffer() {}

      CUDADeviceBuffer(const CUDADeviceBuffer&) = delete;

      CUDADeviceBuffer& operator=(const CUDADeviceBuffer&) = delete;

      CUDADeviceBuffer(CUDADeviceBuffer&& rhs) { _buffer = std::move(rhs._buffer); }

      CUDADeviceBuffer& operator=(CUDADeviceBuffer&& rhs) {
        _buffer = std::move(rhs._buffer);
        return *this;
      }

      virtual T* get(size_t offset = 0) const final { return reinterpret_cast<T*>(_buffer.get()); }

      virtual void upload(const T* src, size_t count, size_t offset = 0) override {
        __message("uploading ", count * sizeof(T), " byte");
        _buffer.upload(src, count * sizeof(T), offset);
      }

      virtual void download(T* dest, size_t count, size_t offset = 0) override {
        __message("downloading ", count * sizeof(T), " byte");
        _buffer.download(dest, count * sizeof(T), offset);
      }

      virtual void uploadAsync(const T* src, size_t count, size_t offset = 0) override {
        _buffer.uploadAsync(src, count * sizeof(T), offset);
      }

      virtual void downloadAsync(T* dest, size_t count, size_t offset = 0) override {
        _buffer.downloadAsync(dest, count * sizeof(T), offset);
      }

      virtual void abandon() override {
        __message("abandoning buffer");
        _buffer.abandon();
      }

    private:
      CUDARawDeviceBuffer _buffer;
    };
  }
}
#endif // PACXX_V2_CUDADEVICEBUFFER_H
