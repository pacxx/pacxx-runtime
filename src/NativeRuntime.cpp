//
// Created by mhaidl on 14/06/16.
//
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include "detail/native/NativeRuntime.h"
#include "detail/common/Exceptions.h"

using namespace llvm;

namespace pacxx
{
  namespace v2
  {
    NativeRuntime::NativeRuntime(unsigned){}
    NativeRuntime::~NativeRuntime(){}

    void NativeRuntime::link(std::unique_ptr<llvm::Module> M) {
      std::string error;
      EngineBuilder builder{std::move(M)};

      builder.setErrorStr(&error);

      builder.setEngineKind(EngineKind::JIT);

      builder.setMCJITMemoryManager(
          std::unique_ptr<RTDyldMemoryManager>(
              static_cast<RTDyldMemoryManager*>(new SectionMemoryManager())));

      _JITEngine = builder.create();
      if (!_JITEngine) {
        throw new common::generic_exception(error);
      }
      _JITEngine->finalizeObject();

    }

    Kernel& NativeRuntime::getKernel(const std::string& name){ throw common::generic_exception("not implemented"); }

    size_t NativeRuntime::getPreferedMemoryAlignment(){ throw common::generic_exception("not implemented");  }


    RawDeviceBuffer* NativeRuntime::allocateRawMemory(size_t bytes) { throw common::generic_exception("not implemented");  }
    void NativeRuntime::deleteRawMemory(RawDeviceBuffer* ptr) { throw common::generic_exception("not implemented"); }
  }
}