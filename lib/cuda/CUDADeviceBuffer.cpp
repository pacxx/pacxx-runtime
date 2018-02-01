//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "pacxx/detail/cuda/CUDADeviceBuffer.h"
#include "pacxx/detail/cuda/CUDARuntime.h"
#include "pacxx/detail/cuda/CUDAErrorDetection.h"
#include <cuda_runtime.h>

namespace pacxx {
namespace v2 {
CUDARawDeviceBuffer::CUDARawDeviceBuffer(size_t size, CUDARuntime *runtime, MemAllocMode mode)
: _size(size), _runtime(runtime), _mode(mode) {
  switch(_mode) {
  case MemAllocMode::Standard:SEC_CUDA_CALL(cudaMalloc((void **) &_buffer, _size));
    break;
  case MemAllocMode::Unified:SEC_CUDA_CALL(cudaMallocManaged((void **) &_buffer, _size));
    break;
  }
  __debug("Allocating ", _size, "b");
  if (_runtime->getProfiler()->enabled())
  {
    count_shadow = _size;
    src_shadow = new char[_size];
  }
}

CUDARawDeviceBuffer::~CUDARawDeviceBuffer() {
  if (_buffer) {
    SEC_CUDA_CALL(cudaFree(_buffer));
    if (_runtime->getProfiler()->enabled())
    {
      if (src_shadow) delete[] src_shadow;
      else __warning("(decon)shadow double clean");
      src_shadow = nullptr;
      offset_shadow = 0;
      count_shadow = 0;
    }
  }
}

CUDARawDeviceBuffer::CUDARawDeviceBuffer(CUDARawDeviceBuffer &&rhs) {
  _buffer = rhs._buffer;
  rhs._buffer = nullptr;
  _size = rhs._size;
  rhs._size = 0;

  if (_runtime->getProfiler()->enabled())
  {
    src_shadow = rhs.src_shadow;
    rhs.src_shadow = nullptr;
    offset_shadow = rhs.offset_shadow;
    rhs.offset_shadow = 0;
    count_shadow = rhs.count_shadow;
    rhs.count_shadow = 0;
  }
}

CUDARawDeviceBuffer &CUDARawDeviceBuffer::operator=(CUDARawDeviceBuffer &&rhs) {
  _buffer = rhs._buffer;
  rhs._buffer = nullptr;
  _size = rhs._size;
  rhs._size = 0;

  if (_runtime->getProfiler()->enabled())
  {
    src_shadow = rhs.src_shadow;
    rhs.src_shadow = nullptr;
    offset_shadow = rhs.offset_shadow;
    rhs.offset_shadow = 0;
    count_shadow = rhs.count_shadow;
    rhs.count_shadow = 0;
  }

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

void CUDARawDeviceBuffer::enshadow() {
  if (_runtime->getProfiler()->enabled())
  {
    __debug("Storing ", count_shadow, "b");
    if (count_shadow) SEC_CUDA_CALL(cudaMemcpy(src_shadow, _buffer + offset_shadow, count_shadow, cudaMemcpyDeviceToHost));
    __debug("Stored ", count_shadow, "b");
  }
}

void CUDARawDeviceBuffer::restore() {
  if (_runtime->getProfiler()->enabled())
  {
    __debug("Restoring ", count_shadow, "b");
    if (count_shadow) SEC_CUDA_CALL(
                          cudaMemcpy(_buffer + offset_shadow, src_shadow, count_shadow, cudaMemcpyHostToDevice));
    __debug("Restored ", count_shadow, "b");
  }
}

void CUDARawDeviceBuffer::copyTo(void *dest) {
  if (!dest)
    __error(__func__, " nullptr arrived, discarding copy");
  if (dest != _buffer)
    SEC_CUDA_CALL(
        cudaMemcpyAsync(dest, _buffer, _size, cudaMemcpyDeviceToDevice));
}
}
}
