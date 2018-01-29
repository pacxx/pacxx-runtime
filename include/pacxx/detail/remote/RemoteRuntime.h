//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "../Runtime.h"
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

class RemoteRuntime : public Runtime {

  friend class RemoteKernel;
  friend class RemoteRawDeviceBuffer;

public:
  static bool classof(const Runtime *rt) {
    return rt->getKind() == RuntimeKind::RK_Remote;
  }

  virtual bool checkSupportedHardware() final { return true; }; 

  RemoteRuntime(const std::string& host, const std::string& port, RuntimeKind rkind, unsigned dev_id);

  virtual ~RemoteRuntime();

  virtual void link(std::unique_ptr<llvm::Module> M) override;

  virtual Kernel &getKernel(const std::string &name) override;

  virtual size_t getPreferedMemoryAlignment() override;

  virtual size_t getPreferedVectorSize(size_t dtype_size) override;

  virtual size_t getPreferedVectorSizeInBytes() override;

  virtual size_t getConcurrentCores() override;

  virtual bool supportsUnifiedAddressing() override;

  virtual std::unique_ptr<RawDeviceBuffer>
  allocateRawMemory(size_t bytes,
                    MemAllocMode mode = MemAllocMode::Standard) override;

  virtual void requestIRTransformation(Kernel &K) override;

  virtual const llvm::Module &getModule() override;

  virtual void synchronize() override;

private:
  std::string send_message(std::string message);

  std::string send_data(const void *data, size_t size);

  std::string read_data(void *ptr, size_t size);

  void connectToDeamon(const std::string &host, const std::string &port);

  void disconnectFromDeamon();

  void createRemoteBackend(Runtime::RuntimeKind kind,
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
