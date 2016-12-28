//
// Created by lars on 07/10/16.
//
#include <detail/native/NativeDeviceBuffer.h>
#include <Executor.h>
#include <malloc.h>

namespace pacxx {
  namespace v2 {
    NativeRawDeviceBuffer::NativeRawDeviceBuffer() : _size(0), _mercy(1), _isHost(false) {}

    void NativeRawDeviceBuffer::allocate(size_t bytes) {
        _buffer = (char*) malloc(bytes);
        if(!_buffer)
            throw new common::generic_exception("buffer allocation failed");
      _size = bytes;
    }

    void NativeRawDeviceBuffer::allocate(size_t bytes, char *host_ptr) {
        _buffer = host_ptr;
        _size = bytes;
        _isHost = true;
    }

    NativeRawDeviceBuffer::~NativeRawDeviceBuffer() {
      if (_buffer && !_isHost) {
        __verbose("isHost", _isHost);
        free(_buffer);
      }
    }

    NativeRawDeviceBuffer::NativeRawDeviceBuffer(NativeRawDeviceBuffer&& rhs) {
      _buffer = rhs._buffer;
      rhs._buffer = nullptr;
      _size = rhs._size;
      rhs._size = 0;
    }

    NativeRawDeviceBuffer& NativeRawDeviceBuffer::operator=(NativeRawDeviceBuffer&& rhs) {
      _buffer = rhs._buffer;
      rhs._buffer = nullptr;
      _size = rhs._size;
      rhs._size = 0;
      return *this;
    }

    void* NativeRawDeviceBuffer::get(size_t offset) const {
      return _buffer + offset;
    }

    void NativeRawDeviceBuffer::upload(const void* src, size_t bytes, size_t offset) {
        std::memcpy(_buffer + offset, src, bytes);
    }

    void NativeRawDeviceBuffer::download(void* dest, size_t bytes, size_t offset) {
        std::memcpy(dest, _buffer + offset, bytes);
    }

    void NativeRawDeviceBuffer::uploadAsync(const void* src, size_t bytes, size_t offset) {
        upload(src, bytes, offset);
    }

    void NativeRawDeviceBuffer::downloadAsync(void* dest, size_t bytes, size_t offset) {
        download(dest, bytes, offset);
    }

    void NativeRawDeviceBuffer::abandon() {
        --_mercy;
        if (_mercy == 0) {
            Executor<NativeRuntime>::Create().freeRaw(*this);
            _buffer = nullptr;
        }
    }

    void NativeRawDeviceBuffer::mercy() {
        ++_mercy;
    }

    void NativeRawDeviceBuffer::copyTo(void* dest)
    {
      if (!dest)
        __error(__func__, "nullptr arrived, discarding copy");
      if (dest != _buffer)
          std::memcpy(dest, _buffer, _size);
    }
  }
}

