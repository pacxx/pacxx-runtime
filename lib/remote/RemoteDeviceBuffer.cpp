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
  _buffer = reinterpret_cast<char*>(_runtime->allocateRemoteMemory(size));
}

RemoteRawDeviceBuffer::~RemoteRawDeviceBuffer() {
  if (_buffer) {
    _runtime->freeRemoteMemory(_buffer);
  }
}

RemoteRawDeviceBuffer::RemoteRawDeviceBuffer(RemoteRawDeviceBuffer &&rhs) {
  _buffer = rhs._buffer;
  rhs._buffer = nullptr;
  _size = rhs._size;
  rhs._size = 0;
}

RemoteRawDeviceBuffer &RemoteRawDeviceBuffer::
operator=(RemoteRawDeviceBuffer &&rhs) {
  _buffer = rhs._buffer;
  rhs._buffer = nullptr;
  _size = rhs._size;
  rhs._size = 0;
  return *this;
}

void *RemoteRawDeviceBuffer::get(size_t offset) const {
  return _buffer + offset;
}

void RemoteRawDeviceBuffer::upload(const void *src, size_t bytes,
                                   size_t offset) {
  _runtime->uploadToRemoteMemory(_buffer + offset, src, bytes);
}

void RemoteRawDeviceBuffer::download(void *dest, size_t bytes, size_t offset) {
  _runtime->downloadFromRemoteMemory(dest, _buffer + offset, bytes);
}

void RemoteRawDeviceBuffer::uploadAsync(const void *src, size_t bytes,
                                        size_t offset) {
  _runtime->uploadToRemoteMemory(_buffer + offset, src, bytes);
}

void RemoteRawDeviceBuffer::downloadAsync(void *dest, size_t bytes,
                                          size_t offset) {
  _runtime->downloadFromRemoteMemory(dest, _buffer + offset, bytes);
}

void RemoteRawDeviceBuffer::copyTo(void *dest) {
  // TODO
}
} // namespace v2
} // namespace pacxx