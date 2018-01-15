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
CUDARawDeviceBuffer::CUDARawDeviceBuffer(
    std::function<void(CUDARawDeviceBuffer&)> deleter,
    MemAllocMode mode)
    : _size(0), _mercy(1), _mode(mode), _deleter(deleter) {}

void CUDARawDeviceBuffer::allocate(size_t bytes) {
  switch(_mode) {
  case MemAllocMode::Standard:SEC_CUDA_CALL(cudaMalloc((void **) &_buffer, bytes));
    break;
  case MemAllocMode::Unified:SEC_CUDA_CALL(cudaMallocManaged((void **) &_buffer, bytes));
    break;
  }
  __debug("Allocating ", bytes, "b");
  _size = bytes;
  count_shadow = bytes;
  src_shadow = new char[bytes];
}

CUDARawDeviceBuffer::~CUDARawDeviceBuffer() {
  if (_buffer) {
    SEC_CUDA_CALL(cudaFree(_buffer));
    if (src_shadow) delete[] src_shadow;
    else __warning("(decon)shadow double clean");
    offset_shadow = 0;
    count_shadow = 0;
  }
}

CUDARawDeviceBuffer::CUDARawDeviceBuffer(CUDARawDeviceBuffer &&rhs) {
  _buffer = rhs._buffer;
  rhs._buffer = nullptr;
  _size = rhs._size;
  rhs._size = 0;
  _mercy = rhs._mercy;
  rhs._mercy = 0;

  src_shadow = rhs.src_shadow;
  rhs.src_shadow = nullptr;
  offset_shadow = rhs.offset_shadow;
  rhs.offset_shadow = 0;
  count_shadow = rhs.count_shadow;
  rhs.count_shadow = 0;
}

CUDARawDeviceBuffer &CUDARawDeviceBuffer::operator=(CUDARawDeviceBuffer &&rhs) {
  _buffer = rhs._buffer;
  rhs._buffer = nullptr;
  _size = rhs._size;
  rhs._size = 0;
  _mercy = rhs._mercy;
  rhs._mercy = 0;

  src_shadow = rhs.src_shadow;
  rhs.src_shadow = nullptr;
  offset_shadow = rhs.offset_shadow;
  rhs.offset_shadow = 0;
  count_shadow = rhs.count_shadow;
  rhs.count_shadow = 0;

  return *this;
}

void *CUDARawDeviceBuffer::get(size_t offset) const { return _buffer + offset; }

void CUDARawDeviceBuffer::upload(const void *src, size_t bytes, size_t offset) {
  __debug("Storing ", bytes, "b");
  if (count_shadow && count_shadow != bytes) __warning("Double upload");
  count_shadow = bytes;
  offset_shadow = offset;
  SEC_CUDA_CALL(
       cudaMemcpy(_buffer + offset, src, bytes, cudaMemcpyHostToDevice));
  SEC_CUDA_CALL(
	  cudaMemcpy(src_shadow, _buffer + offset, bytes, cudaMemcpyDeviceToHost));
  __debug("Stored ", count_shadow, "b");
}

void CUDARawDeviceBuffer::download(void *dest, size_t bytes, size_t offset) {
  SEC_CUDA_CALL(
      cudaMemcpy(dest, _buffer + offset, bytes, cudaMemcpyDeviceToHost));
}

void CUDARawDeviceBuffer::uploadAsync(const void *src, size_t bytes,
                                      size_t offset) {
  __debug("Storing ", bytes, "b");
  if (count_shadow && count_shadow != bytes) __warning("Double upload");
  count_shadow = bytes;
  offset_shadow = offset;
  SEC_CUDA_CALL(
      cudaMemcpyAsync(_buffer + offset, src, bytes, cudaMemcpyHostToDevice));
  SEC_CUDA_CALL(
	  cudaMemcpyAsync(src_shadow, _buffer + offset, bytes, cudaMemcpyDeviceToHost));
  __debug("Stored ", count_shadow, "b");
}

void CUDARawDeviceBuffer::downloadAsync(void *dest, size_t bytes,
                                        size_t offset) {
  SEC_CUDA_CALL(
      cudaMemcpyAsync(dest, _buffer + offset, bytes, cudaMemcpyDeviceToHost));
}

void CUDARawDeviceBuffer::restore() {
  __debug("Restoring ", count_shadow, "b");
  if (count_shadow) SEC_CUDA_CALL(
						cudaMemcpy(_buffer + offset_shadow, src_shadow, count_shadow, cudaMemcpyHostToDevice));
  __debug("Restored ", count_shadow, "b");
}

void CUDARawDeviceBuffer::abandon() {
  --_mercy;
  if (_mercy == 0) {
    _deleter(*this);
    _buffer = nullptr;
    delete[] src_shadow;
    offset_shadow = 0;
    count_shadow = 0;
  }
}

void CUDARawDeviceBuffer::mercy() { ++_mercy; }

void CUDARawDeviceBuffer::copyTo(void *dest) {
  if (!dest)
    __error(__func__, "nullptr arrived, discarding copy");
  if (dest != _buffer)
    SEC_CUDA_CALL(
        cudaMemcpyAsync(dest, _buffer, _size, cudaMemcpyDeviceToDevice));
}
}
}
