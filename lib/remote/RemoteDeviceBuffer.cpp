//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "pacxx/detail/remote/RemoteDeviceBuffer.h"
#include "pacxx/detail/remote/RemoteRuntime.h"

namespace pacxx {
namespace v2 {
RemoteRawDeviceBuffer::RemoteRawDeviceBuffer( size_t size, RemoteRuntime *runtime)
    : _size(size), _runtime(runtime) {
  _buffer = reinterpret_cast<char*>(_runtime->allocateRemoteMemory(_size));
  __debug("Allocating ", _size, "b");
  if (_runtime->getProfiler()->enabled())
  {
    count_shadow = _size;
    src_shadow = new char[_size];
  }
}

RemoteRawDeviceBuffer::~RemoteRawDeviceBuffer() {
  if (_buffer) {
    _runtime->freeRemoteMemory(_buffer);
    if (_runtime->getProfiler()->enabled())
    {
      if (src_shadow) delete[] src_shadow;
      else __warning("(decon)shadow double clean");
      offset_shadow = 0;
      count_shadow = 0;
    }
  }
}

RemoteRawDeviceBuffer::RemoteRawDeviceBuffer(RemoteRawDeviceBuffer &&rhs) {
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

RemoteRawDeviceBuffer &RemoteRawDeviceBuffer::
operator=(RemoteRawDeviceBuffer &&rhs) {
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

void *RemoteRawDeviceBuffer::get(size_t offset) const {
  return _buffer + offset;
}

void RemoteRawDeviceBuffer::upload(const void *src, size_t bytes,
                                   size_t offset) {
  if (_runtime->getProfiler()->enabled())
  {
    __debug("Storing ", bytes, "b");
    if (count_shadow && count_shadow != bytes) __warning("Double upload");
    count_shadow = bytes;
    offset_shadow = offset;
  }
  _runtime->uploadToRemoteMemory(_buffer + offset, src, bytes);
  if (_runtime->getProfiler()->enabled())
  {
    _runtime->downloadFromRemoteMemory(src_shadow, _buffer + offset, bytes);
    __debug("Stored ", count_shadow, "b");
  }
}

void RemoteRawDeviceBuffer::download(void *dest, size_t bytes, size_t offset) {
  _runtime->downloadFromRemoteMemory(dest, _buffer + offset, bytes);
}

void RemoteRawDeviceBuffer::uploadAsync(const void *src, size_t bytes,
                                        size_t offset) {
  upload(src, bytes, offset);
}

void RemoteRawDeviceBuffer::downloadAsync(void *dest, size_t bytes,
                                          size_t offset) {
  download(dest, bytes, offset);
}

void RemoteRawDeviceBuffer::restore() {
  if (_runtime->getProfiler()->enabled())
  {
    __debug("Restoring ", count_shadow, "b");
    if (count_shadow) _runtime->uploadToRemoteMemory(_buffer + offset_shadow, src_shadow, count_shadow);
    __debug("Restored ", count_shadow, "b");
  }
}

void RemoteRawDeviceBuffer::copyTo(void *dest) {
  // TODO
}
} // namespace v2
} // namespace pacxx
