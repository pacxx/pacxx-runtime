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
    NativeRuntime::NativeRuntime(unsigned)
        : _compiler(std::make_unique<CompilerT>()){}

    NativeRuntime::~NativeRuntime() {}

    void NativeRuntime::link(std::unique_ptr<llvm::Module> M) {
      std::string error;

      _compiler->linkInModule(M.get());

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

    void NativeRuntime::initializeMSP(std::unique_ptr <llvm::Module> M) { throw common::generic_exception("not implemented"); }

    void NativeRuntime::evaluateStagedFunctions(Kernel& K) { throw common::generic_exception("not implemented"); }

    void NativeRuntime::requestIRTransformation(Kernel& K) { throw common::generic_exception("not implemented"); };

    const llvm::Module& NativeRuntime::getModule() { throw common::generic_exception("not implemented"); }

    void NativeRuntime::synchronize() { throw common::generic_exception("not implemented"); };

    llvm::legacy::PassManager& NativeRuntime::getPassManager() { return _compiler->getPassManager(); };

  }
}