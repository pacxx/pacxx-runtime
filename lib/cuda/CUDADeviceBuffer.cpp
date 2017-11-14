//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "pacxx/detail/cuda/CUDADeviceBuffer.h"
#include "pacxx/detail/cuda/CUDAErrorDetection.h"
#include <cuda_runtime.h>

namespace pacxx {
namespace v2 {
CUDARawDeviceBuffer::CUDARawDeviceBuffer(std::function<void(CUDARawDeviceBuffer&)> deleter, MemAllocMode mode) : _size(0), _mercy(1), _mode(mode), _deleter(deleter) {}

void CUDARawDeviceBuffer::allocate(size_t bytes) {
  switch(_mode) {
  case MemAllocMode::Standard:SEC_CUDA_CALL(cudaMalloc((void **) &_buffer, bytes));
    break;
  case MemAllocMode::Unified:SEC_CUDA_CALL(cudaMallocManaged((void **) &_buffer, bytes));
    break;
  }
  _size = bytes;
}

CUDARawDeviceBuffer::~CUDARawDeviceBuffer() {
  if (_buffer) {
    SEC_CUDA_CALL(cudaFree(_buffer));
  }
}

CUDARawDeviceBuffer::CUDARawDeviceBuffer(CUDARawDeviceBuffer &&rhs) {
  _buffer = rhs._buffer;
  rhs._buffer = nullptr;
  _size = rhs._size;
  rhs._size = 0;
}

CUDARawDeviceBuffer &CUDARawDeviceBuffer::operator=(CUDARawDeviceBuffer &&rhs) {
  _buffer = rhs._buffer;
  rhs._buffer = nullptr;
  _size = rhs._size;
  rhs._size = 0;
  return *this;
}

void *CUDARawDeviceBuffer::get(size_t offset) const { return _buffer + offset; }

void CUDARawDeviceBuffer::upload(const void *src, size_t bytes, size_t offset) {
  SEC_CUDA_CALL(
      cudaMemcpy(_buffer + offset, src, bytes, cudaMemcpyHostToDevice));
}

void CUDARawDeviceBuffer::download(void *dest, size_t bytes, size_t offset) {
  SEC_CUDA_CALL(
      cudaMemcpy(dest, _buffer + offset, bytes, cudaMemcpyDeviceToHost));
}

void CUDARawDeviceBuffer::uploadAsync(const void *src, size_t bytes,
                                      size_t offset) {
  SEC_CUDA_CALL(
      cudaMemcpyAsync(_buffer + offset, src, bytes, cudaMemcpyHostToDevice));
}

void CUDARawDeviceBuffer::downloadAsync(void *dest, size_t bytes,
                                        size_t offset) {
  SEC_CUDA_CALL(
      cudaMemcpyAsync(dest, _buffer + offset, bytes, cudaMemcpyDeviceToHost));
}

void CUDARawDeviceBuffer::abandon() {
  --_mercy;
  if (_mercy == 0) {
    _deleter(*this);
    _buffer = nullptr;
  }
}

void CUDARawDeviceBuffer::mercy() { ++_mercy; }

void CUDARawDeviceBuffer::copyTo(void *dest) {
  if (!dest)
    __error(__func__, "nullptr arived, discarding copy");
  if (dest != _buffer)
    SEC_CUDA_CALL(
        cudaMemcpyAsync(dest, _buffer, _size, cudaMemcpyDeviceToDevice));
}
}
}