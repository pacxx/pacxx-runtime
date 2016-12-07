//
// Created by mhaidl on 14/06/16.
//
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/PACXXTransforms.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/LinkAllPasses.h>
#include "detail/native/NativeRuntime.h"
#include "detail/common/Exceptions.h"

using namespace llvm;

namespace pacxx
{
  namespace v2
  {

    NativeRuntime::NativeRuntime(unsigned)
        : _compiler(std::make_unique<CompilerT>()), _delayed_compilation(false) {}

    NativeRuntime::~NativeRuntime() {}

    void NativeRuntime::link(std::unique_ptr<llvm::Module> M) {

      _rawM = std::move(M);

      _M.reset(CloneModule(_rawM.get()));
      _M->setDataLayout(_rawM->getDataLayoutStr());

      auto reflect = _M->getFunction("__pacxx_reflect");
      if (!reflect || reflect->getNumUses() == 0) {
        compileAndLink();
      }
      else {
        __verbose("Module contains unresolved calls to __pacxx_reflect. Linking delayed!");
        _CPUMod = _M.get();
        _delayed_compilation = true;
      }
    }

    void NativeRuntime::compileAndLink() {
        _CPUMod  = _compiler->compile(*_M);
        _delayed_compilation = false;
    }

    void NativeRuntime::propagateConstants(NativeKernel &Kernel) {
        KernelConfiguration config = Kernel.getConfiguration();
        std::vector<char> args = Kernel.getHostArguments();

        legacy::PassManager PM = getPassManager();
        PM.add(createPACXXConstantInserterPass(Kernel.getName(), config.threads.x, args));
        PM.add(createSCCPPass());
        PM.add(createDeadCodeEliminationPass());
        PM.run(*_CPUMod);
    }

    Kernel& NativeRuntime::getKernel(const std::string& name){
      auto It = std::find_if(_kernels.begin(), _kernels.end(),
                             [&](const auto &p) { return name == p.first; });
      if (It == _kernels.end()) {
        void *fptr= nullptr;
        if (!_delayed_compilation) {
            fptr = _compiler->getKernelFptr(_CPUMod, name);
            if (!fptr)
                throw common::generic_exception("Kernel function not found in module!");
        }
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
      __verbose("evaluating staged functions");
      if (K.requireStaging()) {
        if (_msp_engine.isDisabled()) return;
        _msp_engine.evaluate(*_rawM->getFunction(K.getName()), K);
      }
    }

    void NativeRuntime::requestIRTransformation(Kernel& K) {
        if (_msp_engine.isDisabled()) return;

        _M.reset(CloneModule(_rawM.get()));
        _M->setDataLayout(_rawM->getDataLayoutStr());
        _msp_engine.transformModule(*_M, K);

        _CPUMod = _compiler->compile(*_M);

        void *fptr= nullptr;
        fptr = _compiler->getKernelFptr(_CPUMod, K.getName().c_str());
        static_cast<NativeKernel &>(K).overrideFptr(fptr);
    }

    const llvm::Module& NativeRuntime::getModule() { return *_CPUMod; }

    void NativeRuntime::synchronize() {};

    llvm::legacy::PassManager& NativeRuntime::getPassManager() { return _compiler->getPassManager(); };

  }
}