//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "../IRRuntime.h"
#include "../msp/MSPEngine.h"
#include "RemoteDeviceBuffer.h"
#include "RemoteKernel.h"
#include "pacxx/detail/common/Exceptions.h"
#include <algorithm>
#include <cstdlib>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <asio.hpp>

namespace pacxx {
namespace v2 {

class RemoteRuntime : public IRRuntime {

  friend class RemoteKernel;
  friend class RemoteRawDeviceBuffer;

public:
  static bool classof(const IRRuntime *rt) {
    return rt->getKind() == RuntimeKind::RK_Remote;
  }

  static bool checkSupportedHardware();

  RemoteRuntime(const std::string& host, const std::string& port, RuntimeKind rkind, unsigned dev_id);

  virtual ~RemoteRuntime();

  virtual void link(std::unique_ptr<llvm::Module> M) override;

  virtual Kernel &getKernel(const std::string &name) override;

  virtual size_t getPreferedMemoryAlignment() override;

  virtual size_t getPreferedVectorSize(size_t dtype_size) override;

  virtual size_t getPreferedVectorSizeInBytes() override;

  virtual size_t getConcurrentCores() override;

  virtual bool supportsUnifiedAddressing() override;

  template <typename T>
  DeviceBuffer<T> *allocateMemory(size_t count, T *host_ptr,
                                  MemAllocMode mode = Standard) {
    auto raw = std::make_unique<RemoteRawDeviceBuffer>(
        [this](RemoteRawDeviceBuffer &buffer) { deleteRawMemory(&buffer); },
        this, mode);
    raw->allocate(count * sizeof(T));
    auto wrapped = new DeviceBuffer<T>(std::move(raw));
    _memory.push_back(std::unique_ptr<DeviceBufferBase<void>>(
        reinterpret_cast<DeviceBufferBase<void> *>(wrapped)));
    if (host_ptr)
      wrapped->upload(host_ptr, count);
    return wrapped;
  }

  template <typename T> DeviceBuffer<T> *translateMemory(T *ptr) {
    auto It =
        std::find_if(_memory.begin(), _memory.end(), [&](const auto &element) {
          return reinterpret_cast<DeviceBuffer<T> *>(element.get())
                     ->get() == ptr;
        });

    if (It != _memory.end())
      return reinterpret_cast<DeviceBuffer<T> *>(It->get());
    else
      throw common::generic_exception(
          "supplied pointer not found in translation list");
  }

  template <typename T> void deleteMemory(DeviceBuffer<T> *ptr) {
    auto It =
        std::find_if(_memory.begin(), _memory.end(),
                     [&](const auto &element) { return element.get() == ptr; });

    if (It != _memory.end())
      _memory.erase(It);
  }

  virtual RawDeviceBuffer *
  allocateRawMemory(size_t bytes,
                    MemAllocMode mode = MemAllocMode::Standard) override;

  virtual void deleteRawMemory(RawDeviceBuffer *ptr) override;

  virtual void requestIRTransformation(Kernel &K) override;

  virtual const llvm::Module &getModule() override;

  virtual void synchronize() override;

private:
  std::string send_message(std::string message);

  std::string send_data(const void *data, size_t size);

  std::string read_data(void *ptr, size_t size);

  void connectToDeamon(const std::string &host, const std::string &port);

  void createRemoteBackend(IRRuntime::RuntimeKind kind,
                           const char *llvm, size_t size);

  void *allocateRemoteMemory(size_t size);

  void freeRemoteMemory(void *ptr);

  void uploadToRemoteMemory(void *dest, const void *src, size_t size);

  void downloadFromRemoteMemory(void *dest, const void *src, size_t size);

  void launchRemoteKernel(const std::string &name, const void *args, size_t size,
                          KernelConfiguration config);

  std::map<std::string, std::unique_ptr<RemoteKernel>> _kernels;
  std::unique_ptr<asio::io_service> _service; 
  std::unique_ptr<asio::ip::tcp::socket> _socket;
  RuntimeKind _kind;
};
} // namespace v2
} // namespace pacxx
