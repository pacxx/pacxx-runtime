//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "pacxx/detail/remote/RemoteRuntime.h"
#include "pacxx/detail/common/Exceptions.h"
#include "pacxx/detail/common/Timing.h"
#include "pacxx/detail/common/transforms/Passes.h"
#include <llvm/IR/Constants.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Vectorize.h>

#include <asio.hpp>

using namespace asio::ip;

using namespace llvm;

namespace pacxx {
namespace v2 {
RemoteRuntime::RemoteRuntime(const std::string &host, const std::string &port,
                             RuntimeKind rkind, unsigned dev_id)
    : IRRuntime(RuntimeKind::RK_Remote), _kind(rkind) {
  connectToDeamon(host, port);
}

RemoteRuntime::~RemoteRuntime() {
  _memory.clear();
  disconnectFromDeamon();
}

void RemoteRuntime::link(std::unique_ptr<llvm::Module> M) {

  _rawM = std::move(M);

  _M = CloneModule(_rawM.get());
  _M->setDataLayout(_rawM->getDataLayoutStr());

  llvm::SmallString<32768> llvm;
  llvm::raw_svector_ostream OS(llvm);

  _M->print(OS, nullptr);

  llvm::legacy::PassManager PM;
  PassManagerBuilder builder;
  builder.OptLevel = 3;
  PM.add(createPACXXCodeGenPrepare());
  builder.populateModulePassManager(PM);
  PM.run(*_M);

  createRemoteBackend(_kind, llvm.data(), llvm.size());
}

Kernel &RemoteRuntime::getKernel(const std::string &name) {
  auto It = std::find_if(_kernels.begin(), _kernels.end(),
                         [&](const auto &p) { return name == p.first; });
  if (It == _kernels.end()) {

    _kernels[name].reset(new RemoteKernel(*this, name));

    return *_kernels[name];
  } else {
    return *It->second;
  }
}

size_t RemoteRuntime::getPreferedMemoryAlignment() {
  return 256; // TODO
}

RawDeviceBuffer *RemoteRuntime::allocateRawMemory(size_t bytes,
                                                  MemAllocMode mode) {
  auto wrapped = allocateMemory<char>(bytes, nullptr, mode);
  return wrapped->getRawBuffer();
}

void RemoteRuntime::deleteRawMemory(RawDeviceBuffer *ptr) {
  auto It = std::find_if(_memory.begin(), _memory.end(), [&](const auto &uptr) {
    return reinterpret_cast<DeviceBuffer<char> *>(uptr.get())->getRawBuffer() ==
           ptr;
  });
  if (It != _memory.end())
    _memory.erase(It);
  else
    __error("ptr to delete not found");
}

std::string RemoteRuntime::send_message(std::string message) {
  std::string answer;
  asio::streambuf buffer;
  asio::write(*_socket, asio::buffer(message + "\r"));
  std::cout << message << " -> ";
  asio::read_until(*_socket, buffer, '\r');
  std::istream is(&buffer);
  is >> answer;
  std::cout << answer << std::endl;
  return answer;
}

std::string RemoteRuntime::send_data(const void *data, size_t size) {
  std::string answer;
  asio::streambuf buffer;
  asio::write(*_socket, asio::buffer(data, size));
  std::cout << data << " -> ";
  asio::read_until(*_socket, buffer, '\r');
  std::istream is(&buffer);
  is >> answer;
  std::cout << answer << std::endl;
  return answer;
}

std::string RemoteRuntime::read_data(void *ptr, size_t size) {
  asio::read(*_socket, asio::buffer(ptr, size), asio::transfer_exactly(size));
  return send_message("ACK");
}

void RemoteRuntime::connectToDeamon(const std::string &host,
                                    const std::string &port) {
  _service.reset(new asio::io_service());
  tcp::resolver resolver(*_service);
  tcp::resolver::query query(host, port);
  tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);

  _socket.reset(new asio::ip::tcp::socket(*_service));
  asio::connect(*_socket, endpoint_iterator);

  send_message("HELLO");
}

void RemoteRuntime::disconnectFromDeamon() {
  send_message("BYE");
  _socket.reset();
  _service.reset();
}

void RemoteRuntime::createRemoteBackend(pacxx::v2::IRRuntime::RuntimeKind kind,
                                        const char *llvm, size_t size) {
  std::string rtName;
  using namespace pacxx::v2;
  switch (kind) {
  case IRRuntime::RK_CUDA:
    rtName = "CUDA";
    break;
  case IRRuntime::RK_Native:
    rtName = "NATIVE";
    break;
  case IRRuntime::RK_HIP:
    rtName = "HIP";
    break;
  default:
    throw common::generic_exception("cannot create this remote backend");
  }

  send_message(rtName);
  send_message("LLVM");
  send_message(std::to_string(size));
  send_data(llvm, size);
}

void *RemoteRuntime::allocateRemoteMemory(size_t size) {
  send_message("ALLOC");
  return reinterpret_cast<void *>(
      std::stoul(send_message(std::to_string(size))));
}

void RemoteRuntime::freeRemoteMemory(void *ptr) {
  send_message("FREE");
  send_message(std::to_string(reinterpret_cast<uint64_t>(ptr)));
}

void RemoteRuntime::uploadToRemoteMemory(void *dest, const void *src,
                                         size_t size) {
  send_message("UPLOAD");
  send_message(std::to_string(reinterpret_cast<uint64_t>(dest)));
  send_message(std::to_string(size));
  send_data(src, size);
}

void RemoteRuntime::downloadFromRemoteMemory(void *dest, const void *src,
                                             size_t size) {
  send_message("DOWNLOAD");
  send_message(std::to_string(reinterpret_cast<uint64_t>(src)));
  send_message(std::to_string(size));
  auto answer = read_data(dest, size);
}

void RemoteRuntime::launchRemoteKernel(const std::string &name,
                                       const void *args, size_t size,
                                       KernelConfiguration config) {
  send_message("LAUNCH");
  send_message(name);
  send_message(std::to_string(config.blocks.x));
  send_message(std::to_string(config.blocks.y));
  send_message(std::to_string(config.blocks.z));
  send_message(std::to_string(config.threads.x));
  send_message(std::to_string(config.threads.y));
  send_message(std::to_string(config.threads.z));
  send_message(std::to_string(size));
  send_data(args, size);
}

void RemoteRuntime::synchronize() {} // TODO

size_t RemoteRuntime::getPreferedVectorSize(size_t dtype_size) {
  return 1; // TODO
}

size_t RemoteRuntime::getPreferedVectorSizeInBytes() {
  return 8; // TODO
}

size_t RemoteRuntime::getConcurrentCores() {
  return 1; // TODO
}

bool RemoteRuntime::supportsUnifiedAddressing() { return false; }

const llvm::Module &RemoteRuntime::getModule() { return *_M; }

void RemoteRuntime::requestIRTransformation(Kernel &K) {}

} // namespace v2
} // namespace pacxx
