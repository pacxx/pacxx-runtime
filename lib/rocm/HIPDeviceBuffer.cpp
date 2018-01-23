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
HIPRawDeviceBuffer::HIPRawDeviceBuffer(
    std::function<void(HIPRawDeviceBuffer&)> deleter,
    HIPRuntime *runtime, MemAllocMode mode)
    : _size(0), _mercy(1), _mode(mode), _deleter(deleter), _runtime(runtime) {}

void HIPRawDeviceBuffer::allocate(size_t bytes) {
  switch(_mode) {
  case MemAllocMode::Standard:SEC_HIP_CALL(hipMalloc((void **) &_buffer, bytes));
    break;
  case MemAllocMode::Unified:
    break;
  }
  __debug("Allocating ", bytes, "b");
  _size = bytes;
  if (_runtime->getProfiler()->enabled())
  {
    count_shadow = bytes;
    src_shadow = new char[bytes];
  }
}

HIPRawDeviceBuffer::~HIPRawDeviceBuffer() {
  if (_buffer) {
    SEC_HIP_CALL(hipFree(_buffer));
    if (_runtime->getProfiler()->enabled())
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
  _mercy = rhs._mercy;
  rhs._mercy = 0;

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

HIPRawDeviceBuffer &HIPRawDeviceBuffer::operator=(HIPRawDeviceBuffer &&rhs) {
  _buffer = rhs._buffer;
  rhs._buffer = nullptr;
  _size = rhs._size;
  rhs._size = 0;
  _mercy = rhs._mercy;
  rhs._mercy = 0;

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

void *HIPRawDeviceBuffer::get(size_t offset) const { return _buffer + offset; }

void HIPRawDeviceBuffer::upload(const void *src, size_t bytes, size_t offset) {
  if (_runtime->getProfiler()->enabled())
  {
    __debug("Storing ", bytes, "b");
    if (count_shadow && count_shadow != bytes) __warning("Double upload");
    count_shadow = bytes;
    offset_shadow = offset;
  }
  SEC_HIP_CALL(
      hipMemcpy(_buffer + offset, src, bytes, hipMemcpyHostToDevice));
  if (_runtime->getProfiler()->enabled())
  {
    SEC_HIP_CALL(
        hipMemcpy(src_shadow, _buffer + offset, bytes, hipMemcpyDeviceToHost));
    __debug("Stored ", count_shadow, "b");
  }
}

void HIPRawDeviceBuffer::download(void *dest, size_t bytes, size_t offset) {
  SEC_HIP_CALL(
      hipMemcpy(dest, _buffer + offset, bytes, hipMemcpyDeviceToHost));
}

void HIPRawDeviceBuffer::uploadAsync(const void *src, size_t bytes,
                                      size_t offset) {
  if (_runtime->getProfiler()->enabled())
  {
    __debug("Storing ", bytes, "b");
    if (count_shadow && count_shadow != bytes) __warning("Double upload");
    count_shadow = bytes;
    offset_shadow = offset;
  }
  SEC_HIP_CALL(
      hipMemcpyAsync(_buffer + offset, src, bytes, hipMemcpyHostToDevice));
  if (_runtime->getProfiler()->enabled())
  {
    SEC_HIP_CALL(
      hipMemcpyAsync(src_shadow, _buffer + offset, bytes, hipMemcpyDeviceToHost));
    __debug("Stored ", count_shadow, "b");
  }
}

void HIPRawDeviceBuffer::downloadAsync(void *dest, size_t bytes,
                                        size_t offset) {
  SEC_HIP_CALL(
      hipMemcpyAsync(dest, _buffer + offset, bytes, hipMemcpyDeviceToHost));
}

void HIPRawDeviceBuffer::restore() {
  if (_runtime->getProfiler()->enabled())
  {
    __debug("Restoring ", count_shadow, "b");
    if (count_shadow) SEC_HIP_CALL(
              hipMemcpy(_buffer + offset_shadow, src_shadow, count_shadow, hipMemcpyHostToDevice));
    __debug("Restored ", count_shadow, "b");
  }
}

void HIPRawDeviceBuffer::abandon() {
  --_mercy;
  if (_mercy == 0) {
    _deleter(*this);
    _buffer = nullptr;
    if (_runtime->getProfiler()->enabled())
    {
      delete[] src_shadow;
      offset_shadow = 0;
      count_shadow = 0;
    }
  }
}

void HIPRawDeviceBuffer::mercy() { ++_mercy; }

void HIPRawDeviceBuffer::copyTo(void *dest) {
  if (!dest)
    __error(__func__, "nullptr arrived, discarding copy");
  if (dest != _buffer)
    SEC_HIP_CALL(
        hipMemcpyAsync(dest, _buffer, _size, hipMemcpyDeviceToDevice));
}
}
}
