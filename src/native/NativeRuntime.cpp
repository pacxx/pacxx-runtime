//
// Created by mhaidl on 14/06/16.
//
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include "detail/native/NativeRuntime.h"
#include "detail/common/Exceptions.h"

using namespace llvm;

namespace pacxx
{
  namespace v2
  {

    NativeRuntime::NativeRuntime(unsigned)
        : _compiler(std::make_unique<CompilerT>()) {}

    NativeRuntime::~NativeRuntime() {}

    void NativeRuntime::link(std::unique_ptr<llvm::Module> M) {

      _M = std::move(M);

      _CPUMod = _compiler->compile(*_M);

    }

    Kernel& NativeRuntime::getKernel(const std::string& name){
      auto It = std::find_if(_kernels.begin(), _kernels.end(),
                             [&](const auto &p) { return name == p.first; });
      if (It == _kernels.end()) {
        void *fptr= nullptr;
        fptr = _compiler->getKernelFptr(_CPUMod, name);
        if (!fptr)
          throw common::generic_exception("Kernel function not found in module!");
        auto kernel = new NativeKernel(*this, fptr);
        kernel->setName(name);
        _kernels[name].reset(kernel);

        return *kernel;
      } else {
        return *It->second;
      }
    }

    size_t NativeRuntime::getPreferedMemoryAlignment(){ return _CPUMod->getDataLayout().getPointerABIAlignment(); }

    RawDeviceBuffer* NativeRuntime::allocateRawMemory(size_t bytes) {
      NativeRawDeviceBuffer rawBuffer;
      rawBuffer.allocate(bytes);
      auto wrapped = new NativeDeviceBuffer<char>(std::move(rawBuffer));
      _memory.push_back(std::unique_ptr<DeviceBufferBase>(
              static_cast<DeviceBufferBase *>(wrapped)));
      return wrapped->getRawBuffer();
    }

    void NativeRuntime::deleteRawMemory(RawDeviceBuffer* ptr) {
      auto It = std::find_if(_memory.begin(), _memory.end(), [&](const auto &uptr) {
          return static_cast<NativeDeviceBuffer<char> *>(uptr.get())->getRawBuffer() == ptr;
      });
      if(It != _memory.end())
        _memory.erase(It);
      else
          __error("ptr to delete not found");
    }

    void NativeRuntime::initializeMSP(std::unique_ptr <llvm::Module> M) {
      if (!_msp_engine.isDisabled()) return;
      _msp_engine.initialize(std::move(M));
    }

    void NativeRuntime::evaluateStagedFunctions(Kernel& K) {
      if (K.requireStaging()) {
        if (_msp_engine.isDisabled()) return;
        _msp_engine.evaluate(*_CPUMod->getFunction(K.getName()), K);
      }
    }

    void NativeRuntime::requestIRTransformation(Kernel& K) { throw common::generic_exception("not implemented"); }

    const llvm::Module& NativeRuntime::getModule() { return *_CPUMod; }

    void NativeRuntime::runOnThread(void* fptr, size_t bidx, size_t bidy, size_t bidz, size_t max_x, size_t max_y,
                                    size_t max_z, char* args) {
      // The kernel wrapper always has this function signature.
      // The kernel args are constructed from the char buffer
      auto functor = reinterpret_cast<void (*) (int, int, int,
                                                int, int, int, char*)>(fptr);

      _threadPool.async(functor, bidx, bidy, bidz, max_x, max_y, max_z, args);
    }

    void NativeRuntime::synchronize() {};

    llvm::legacy::PassManager& NativeRuntime::getPassManager() { return _compiler->getPassManager(); };

  }
}