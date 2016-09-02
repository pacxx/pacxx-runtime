//
// Created by mhaidl on 10/08/16.
//
#include <detail/cuda/CUDADeviceBuffer.h>

#include <Executor.h>


namespace pacxx {
  namespace v2 {
    CUDARawDeviceBuffer::CUDARawDeviceBuffer() : _size(0), _mercy(1) {
    }

    void CUDARawDeviceBuffer::allocate(size_t bytes) {
      SEC_CUDA_CALL(cudaMalloc((void**) &_buffer, bytes));
      _size = bytes;
    }

    CUDARawDeviceBuffer::~CUDARawDeviceBuffer() {
      if (_buffer) {
        SEC_CUDA_CALL(cudaFree(_buffer));
     //   __message("freeing buffer");
      }
    }

    CUDARawDeviceBuffer::CUDARawDeviceBuffer(CUDARawDeviceBuffer&& rhs) {
      _buffer = rhs._buffer;
      rhs.
          _buffer = nullptr;
    }

    CUDARawDeviceBuffer& CUDARawDeviceBuffer::operator=(CUDARawDeviceBuffer&& rhs) {
      _buffer = rhs._buffer;
      rhs._buffer = nullptr;
      rhs._size = 0;
      return *this;
    }

    void* CUDARawDeviceBuffer::get(size_t offset) const {
      return _buffer + offset;
    }

    void CUDARawDeviceBuffer::upload(const void* src, size_t bytes, size_t offset) {
      SEC_CUDA_CALL(cudaMemcpy(_buffer + offset, src, bytes, cudaMemcpyHostToDevice));
    }

    void CUDARawDeviceBuffer::download(void* dest, size_t bytes, size_t offset) {
      SEC_CUDA_CALL(cudaMemcpy(dest, _buffer + offset, bytes, cudaMemcpyDeviceToHost));
    }

    void CUDARawDeviceBuffer::uploadAsync(const void* src, size_t bytes, size_t offset) {
      SEC_CUDA_CALL(cudaMemcpyAsync(_buffer + offset, src, bytes, cudaMemcpyHostToDevice));
    }

    void CUDARawDeviceBuffer::downloadAsync(void* dest, size_t bytes, size_t offset) {
      SEC_CUDA_CALL(
          cudaMemcpyAsync(dest, _buffer + offset, bytes, cudaMemcpyDeviceToHost));
    }

    void CUDARawDeviceBuffer::abandon() {
        --_mercy;
        if (_mercy == 0) {
            Executor<CUDARuntime>::Create().freeRaw(*this);
            _buffer = nullptr;
        }
    }

    void CUDARawDeviceBuffer::mercy() {
        ++_mercy;
    }

    void CUDARawDeviceBuffer::copyTo(void* dest)
    {
      SEC_CUDA_CALL(cudaMemcpy(dest, _buffer, _size, cudaMemcpyDeviceToDevice));
    }
  }
}