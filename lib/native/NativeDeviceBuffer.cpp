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
NativeRawDeviceBuffer::NativeRawDeviceBuffer(std::function<void(NativeRawDeviceBuffer&)> deleter)
    : _size(0), _mercy(1), _isHost(false), _deleter(deleter){}

void NativeRawDeviceBuffer::allocate(size_t bytes, unsigned padding) {

  auto padSize = [=](size_t bytes, unsigned vf) {
    if (vf == 0)
      return bytes;

    int remainder = bytes % vf;
    return bytes + vf - remainder; // pad after and before memory
  };

  auto total = padSize(bytes, padding);

  __verbose("allocating padded: ", bytes, " ", padSize(bytes, padding), " ", padding);

  _buffer = (char *) malloc(total);
  if (!_buffer)
    throw new common::generic_exception("buffer allocation failed");

  _size = bytes;
}

void NativeRawDeviceBuffer::allocate(size_t bytes, char *host_ptr) {
  __verbose("allocating host buffer");
  _buffer = host_ptr;
  _size = bytes;
  _isHost = true;
}

NativeRawDeviceBuffer::~NativeRawDeviceBuffer() {
  __verbose("deleting buffer");
  if (_buffer && !_isHost) {
    free(_buffer);
  }
}

NativeRawDeviceBuffer::NativeRawDeviceBuffer(NativeRawDeviceBuffer &&rhs) {
  _buffer = rhs._buffer;
  rhs._buffer = nullptr;
  _size = rhs._size;
  rhs._size = 0;
  _isHost = rhs._isHost;
  rhs._isHost = false;
}

NativeRawDeviceBuffer &NativeRawDeviceBuffer::
operator=(NativeRawDeviceBuffer &&rhs) {
  _buffer = rhs._buffer;
  rhs._buffer = nullptr;
  _size = rhs._size;
  rhs._size = 0;
  _isHost = rhs._isHost;
  rhs._isHost = false;
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

void NativeRawDeviceBuffer::abandon() {
  --_mercy;
  if (_mercy == 0) {
    _deleter(*this);

    _buffer = nullptr;
  }
}

void NativeRawDeviceBuffer::mercy() { ++_mercy; }

void NativeRawDeviceBuffer::copyTo(void *dest) {
  if (!dest)
    __error(__func__, "nullptr arrived, discarding copy");
  if (dest != _buffer)
    std::memcpy(dest, _buffer, _size);
}
}
}
