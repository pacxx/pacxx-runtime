//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "pacxx/detail/native/NativeDeviceBuffer.h"
#include "pacxx/detail/native/NativeRuntime.h"
#include "pacxx/detail/common/Exceptions.h"
#include <cstdlib>
#include <cstring>

namespace pacxx {
namespace v2 {
NativeRawDeviceBuffer::NativeRawDeviceBuffer(size_t size, unsigned padding, NativeRuntime *runtime)
    : _size(size), _runtime(runtime) {
  auto padSize = [=](size_t bytes, unsigned vf) {
    if (vf == 0)
      return bytes;

    int remainder = bytes % vf;
    return bytes + vf - remainder; // pad after and before memory
  };

  auto total = padSize(_size, padding);

  __verbose("allocating padded: ", _size, " ", padSize(_size, padding), " ", padding);

  _buffer = (char *) malloc(total);
  if (!_buffer)
    throw new common::generic_exception("buffer allocation failed");
  __debug("Allocating ", _size, "b");
  if (auto profiler = _runtime->getProfiler())
    if (profiler->enabled()) {
      count_shadow = _size;
      src_shadow = new char[_size];
    }
}

NativeRawDeviceBuffer::~NativeRawDeviceBuffer() {
  __verbose("deleting buffer");
  if (_buffer) {
    free(_buffer);
    if (auto profiler = _runtime->getProfiler())
      if (profiler->enabled()) {
        if (src_shadow) delete[] src_shadow;
        else __warning("(decon)shadow double clean");
        offset_shadow = 0;
        count_shadow = 0;
      }
  }
}

NativeRawDeviceBuffer::NativeRawDeviceBuffer(NativeRawDeviceBuffer &&rhs) {
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

NativeRawDeviceBuffer &NativeRawDeviceBuffer::
operator=(NativeRawDeviceBuffer &&rhs) {
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

void *NativeRawDeviceBuffer::get(size_t offset) const {
  return _buffer + offset;
}

void NativeRawDeviceBuffer::upload(const void *src, size_t bytes,
                                   size_t offset) {
  __verbose("uploading ", bytes, " bytes");
  std::memcpy(_buffer + offset, src, bytes);
}

void NativeRawDeviceBuffer::download(void *dest, size_t bytes, size_t offset) {
  __verbose("downloading ", bytes, " bytes");
  std::memcpy(dest, _buffer + offset, bytes);
}

void NativeRawDeviceBuffer::uploadAsync(const void *src, size_t bytes,
                                        size_t offset) {
  upload(src, bytes, offset);
}

void NativeRawDeviceBuffer::downloadAsync(void *dest, size_t bytes,
                                          size_t offset) {
  download(dest, bytes, offset);
}

void NativeRawDeviceBuffer::enshadow() {
  if (_runtime->getProfiler()->enabled())
  {
    __debug("Storing ", count_shadow, "b");
    if (count_shadow) std::memcpy(src_shadow, _buffer + offset_shadow, count_shadow);
    __debug("Stored ", count_shadow, "b");
  }
}

void NativeRawDeviceBuffer::restore() {
  if (_runtime->getProfiler()->enabled())
  {
    __debug("Restoring ", count_shadow, "b");
    if (count_shadow) std::memcpy(_buffer + offset_shadow, src_shadow, count_shadow);
    __debug("Restored ", count_shadow, "b");
  }
}

void NativeRawDeviceBuffer::copyTo(void *dest) {
  if (!dest)
    __error(__func__, " nullptr arrived, discarding copy");
  if (dest != _buffer)
    std::memcpy(dest, _buffer, _size);
}
}
}
