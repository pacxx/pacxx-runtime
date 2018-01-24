//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "pacxx/detail/native/NativeDeviceBuffer.h"
#include "pacxx/detail/common/Exceptions.h"
#include <cstdlib>
#include <cstring>

namespace pacxx {
namespace v2 {
NativeRawDeviceBuffer::NativeRawDeviceBuffer(size_t size, unsigned padding)
    : _size(size) {
  auto padSize = [=](size_t bytes, unsigned vf) {
    if (vf == 0)
      return bytes;

    int remainder = bytes % vf;
    return bytes + vf - remainder; // pad after and before memory
  };

  auto total = padSize(size, padding);

  __verbose("allocating padded: ", size, " ", padSize(size, padding), " ", padding);

  _buffer = (char *) malloc(total);
  if (!_buffer)
    throw new common::generic_exception("buffer allocation failed");
}

NativeRawDeviceBuffer::~NativeRawDeviceBuffer() {
  __verbose("deleting buffer");
  if (_buffer) {
    free(_buffer);
  }
}

NativeRawDeviceBuffer::NativeRawDeviceBuffer(NativeRawDeviceBuffer &&rhs) {
  _buffer = rhs._buffer;
  rhs._buffer = nullptr;
  _size = rhs._size;
  rhs._size = 0;
}

NativeRawDeviceBuffer &NativeRawDeviceBuffer::
operator=(NativeRawDeviceBuffer &&rhs) {
  _buffer = rhs._buffer;
  rhs._buffer = nullptr;
  _size = rhs._size;
  rhs._size = 0;
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

void NativeRawDeviceBuffer::copyTo(void *dest) {
  if (!dest)
    __error(__func__, "nullptr arrived, discarding copy");
  if (dest != _buffer)
    std::memcpy(dest, _buffer, _size);
}
}
}
