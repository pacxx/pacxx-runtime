//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "pacxx/detail/rocm/HIPDeviceBuffer.h"
#include "pacxx/detail/rocm/HIPErrorDetection.h"

#include <hip/hip_runtime.h>

namespace pacxx {
namespace v2 {
HIPRawDeviceBuffer::HIPRawDeviceBuffer(size_t size) 
: _size(size) {
  SEC_HIP_CALL(hipMalloc((void **) &_buffer, _size)); 
}

HIPRawDeviceBuffer::~HIPRawDeviceBuffer() {
  if (_buffer) {
    SEC_HIP_CALL(hipFree(_buffer));
  }
}

HIPRawDeviceBuffer::HIPRawDeviceBuffer(HIPRawDeviceBuffer &&rhs) {
  _buffer = rhs._buffer;
  rhs._buffer = nullptr;
  _size = rhs._size;
  rhs._size = 0;
}

HIPRawDeviceBuffer &HIPRawDeviceBuffer::operator=(HIPRawDeviceBuffer &&rhs) {
  _buffer = rhs._buffer;
  rhs._buffer = nullptr;
  _size = rhs._size;
  rhs._size = 0;
  return *this;
}

void *HIPRawDeviceBuffer::get(size_t offset) const { return _buffer + offset; }

void HIPRawDeviceBuffer::upload(const void *src, size_t bytes, size_t offset) {
  SEC_HIP_CALL(
      hipMemcpy(_buffer + offset, src, bytes, hipMemcpyHostToDevice));
}

void HIPRawDeviceBuffer::download(void *dest, size_t bytes, size_t offset) {
  SEC_HIP_CALL(
      hipMemcpy(dest, _buffer + offset, bytes, hipMemcpyDeviceToHost));
}

void HIPRawDeviceBuffer::uploadAsync(const void *src, size_t bytes,
                                      size_t offset) {
  SEC_HIP_CALL(
      hipMemcpyAsync(_buffer + offset, src, bytes, hipMemcpyHostToDevice));
}

void HIPRawDeviceBuffer::downloadAsync(void *dest, size_t bytes,
                                        size_t offset) {
  SEC_HIP_CALL(
      hipMemcpyAsync(dest, _buffer + offset, bytes, hipMemcpyDeviceToHost));
}

void HIPRawDeviceBuffer::copyTo(void *dest) {
  if (!dest)
    __error(__func__, "nullptr arived, discarding copy");
  if (dest != _buffer)
    SEC_HIP_CALL(
        hipMemcpyAsync(dest, _buffer, _size, hipMemcpyDeviceToDevice));
}
}
}
