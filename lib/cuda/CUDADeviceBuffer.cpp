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
CUDARawDeviceBuffer::CUDARawDeviceBuffer(size_t size, MemAllocMode mode) 
: _size(size), _mode(mode) {
  switch(_mode) {
  case MemAllocMode::Standard:SEC_CUDA_CALL(cudaMalloc((void **) &_buffer, _size));
    break;
  case MemAllocMode::Unified:SEC_CUDA_CALL(cudaMallocManaged((void **) &_buffer, _size));
    break;
  }
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

void CUDARawDeviceBuffer::copyTo(void *dest) {
  if (!dest)
    __error(__func__, " nullptr arived, discarding copy");
  if (dest != _buffer)
    SEC_CUDA_CALL(
        cudaMemcpyAsync(dest, _buffer, _size, cudaMemcpyDeviceToDevice));
}
}
}