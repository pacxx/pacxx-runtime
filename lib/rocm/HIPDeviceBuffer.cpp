//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "pacxx/detail/rocm/HIPDeviceBuffer.h"
#include "pacxx/detail/rocm/HIPRuntime.h"
#include "pacxx/detail/rocm/HIPErrorDetection.h"

#include <hip/hip_runtime.h>

namespace pacxx {
namespace v2 {
HIPRawDeviceBuffer::HIPRawDeviceBuffer(size_t size, HIPRuntime *runtime)
: _size(size), _runtime(runtime) {
  SEC_HIP_CALL(hipMalloc((void **) &_buffer, _size));
  __debug("Allocating ", _size, "b");
  if (_runtime->getProfiler() && _runtime->getProfiler()->enabled())
  {
    count_shadow = _size;
    src_shadow = new char[_size];
  }
}

HIPRawDeviceBuffer::~HIPRawDeviceBuffer() {
  if (_buffer) {
    SEC_HIP_CALL(hipFree(_buffer));
    if (_runtime->getProfiler() && _runtime->getProfiler()->enabled())
    {
      if (src_shadow) delete[] src_shadow;
      else __warning("(decon)shadow double clean");
      offset_shadow = 0;
      count_shadow = 0;
    }
  }
}

HIPRawDeviceBuffer::HIPRawDeviceBuffer(HIPRawDeviceBuffer &&rhs) {
  _buffer = rhs._buffer;
  rhs._buffer = nullptr;
  _size = rhs._size;
  rhs._size = 0;

  if (_runtime->getProfiler() && _runtime->getProfiler()->enabled())
  {
    src_shadow = rhs.src_shadow;
    rhs.src_shadow = nullptr;
    offset_shadow = rhs.offset_shadow;
    rhs.offset_shadow = 0;
    count_shadow = rhs.count_shadow;
    rhs.count_shadow = 0;
  }
}

HIPRawDeviceBuffer &HIPRawDeviceBuffer::operator=(HIPRawDeviceBuffer &&rhs) {
  _buffer = rhs._buffer;
  rhs._buffer = nullptr;
  _size = rhs._size;
  rhs._size = 0;

  if (_runtime->getProfiler() && _runtime->getProfiler()->enabled())
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

void HIPRawDeviceBuffer::enshadow() {
  if (_runtime->getProfiler() && _runtime->getProfiler()->enabled())
  {
    __debug("Storing ", count_shadow, "b");
    if (count_shadow) SEC_HIP_CALL(
      hipMemcpyAsync(src_shadow, _buffer + offset_shadow, count_shadow, hipMemcpyDeviceToHost));
    __debug("Stored ", count_shadow, "b");
  }
}

void HIPRawDeviceBuffer::restore() {
  if (_runtime->getProfiler() && _runtime->getProfiler()->enabled())
  {
    __debug("Restoring ", count_shadow, "b");
    if (count_shadow) SEC_HIP_CALL(
              hipMemcpy(_buffer + offset_shadow, src_shadow, count_shadow, hipMemcpyHostToDevice));
    __debug("Restored ", count_shadow, "b");
  }
}

void HIPRawDeviceBuffer::copyTo(void *dest) {
  if (!dest)
    __error(__func__, " nullptr arrived, discarding copy");
  if (dest != _buffer)
    SEC_HIP_CALL(
        hipMemcpyAsync(dest, _buffer, _size, hipMemcpyDeviceToDevice));
}
}
}
