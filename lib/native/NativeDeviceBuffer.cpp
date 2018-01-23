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
NativeRawDeviceBuffer::NativeRawDeviceBuffer(
    std::function<void(NativeRawDeviceBuffer&)> deleter,
    NativeRuntime *runtime)
    : _size(0), _mercy(1), _isHost(false), _deleter(deleter), _runtime(runtime) {}

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
  __debug("Allocating ", bytes, "b");
  _size = bytes;
  if (_runtime->getProfiler()->enabled())
  {
    count_shadow = bytes;
    src_shadow = new char[bytes];
  }
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
    if (_runtime->getProfiler()->enabled())
    {
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

  _isHost = rhs._isHost;
  rhs._isHost = false;
}

NativeRawDeviceBuffer &NativeRawDeviceBuffer::
operator=(NativeRawDeviceBuffer &&rhs) {
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
  if (_runtime->getProfiler()->enabled())
  {
    __debug("Storing ", bytes, "b");
    if (count_shadow && count_shadow != bytes) __warning("Double upload");
    count_shadow = bytes;
    offset_shadow = offset;
  }
  std::memcpy(_buffer + offset, src, bytes);
  if (_runtime->getProfiler()->enabled())
  {
    std::memcpy(src_shadow, _buffer + offset, bytes);
    __debug("Stored ", count_shadow, "b");
  }
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

void NativeRawDeviceBuffer::restore() {
  if (_runtime->getProfiler()->enabled())
  {
    __debug("Restoring ", count_shadow, "b");
    if (count_shadow) std::memcpy(_buffer + offset_shadow, src_shadow, count_shadow);
    __debug("Restored ", count_shadow, "b");
  }
}

void NativeRawDeviceBuffer::abandon() {
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

void NativeRawDeviceBuffer::mercy() { ++_mercy; }

void NativeRawDeviceBuffer::copyTo(void *dest) {
  if (!dest)
    __error(__func__, "nullptr arrived, discarding copy");
  if (dest != _buffer)
    std::memcpy(dest, _buffer, _size);
}
}
}
