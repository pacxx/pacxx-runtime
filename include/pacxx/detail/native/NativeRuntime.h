//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef PACXX_V2_NATIVERUNTIME_H
#define PACXX_V2_NATIVERUNTIME_H

#include "../Runtime.h"
#include "NativeBackend.h"
#include "NativeDeviceBuffer.h"
#include "NativeKernel.h"
#include "pacxx/detail/common/Exceptions.h"
#include "pacxx/detail/msp/MSPEngine.h"
#include <list>
#include <map>
#include <memory>
#include <string>
#include <thread>

namespace pacxx {
namespace v2 {

class NativeRuntime : public Runtime {
public:
  using CompilerT = NativeBackend;

  static bool classof(const Runtime *rt) {
    return rt->getKind() == RuntimeKind::RK_Native;
  }

  NativeRuntime(unsigned dev_id);
  virtual ~NativeRuntime();

  virtual void link(std::unique_ptr<llvm::Module> M) override;

  virtual Kernel &getKernel(const std::string &name) override;

  virtual size_t getPreferedMemoryAlignment() override;

  virtual size_t getPreferedVectorSize(size_t dtype_size) override;

  virtual size_t getPreferedVectorSizeInBytes() override;

  virtual size_t getConcurrentCores() override;

  virtual bool supportsUnifiedAddressing() override;

  virtual std::unique_ptr<RawDeviceBuffer> allocateRawMemory(size_t bytes, MemAllocMode mode = Standard) override;

  //virtual void deleteRawMemory(RawDeviceBuffer *ptr) override;

  virtual void requestIRTransformation(Kernel &K) override;

  virtual void synchronize() override;

private:
  void compileAndLink();

private:
  llvm::Module *_CPUMod;

  std::unique_ptr<CompilerT> _compiler;
  std::map<std::string, std::unique_ptr<NativeKernel>> _kernels;
  bool _delayed_compilation;

};
}
}

#endif // PACXX_V2_NATIVERUNTIME_H
