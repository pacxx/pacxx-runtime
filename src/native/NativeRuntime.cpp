//
// Created by mhaidl on 14/06/16.
//
#include "pacxx/detail/native/NativeRuntime.h"
#include "pacxx/detail/common/Exceptions.h"
#include "pacxx/detail/common/Timing.h"
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/LinkAllPasses.h>
#include <llvm/Transforms/PACXXTransforms.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils/Cloning.h>
#ifndef PACXX_DISABLE_TBB
#include <tbb/task_scheduler_init.h>
#endif
#include <thread>

using namespace llvm;

namespace pacxx {
namespace v2 {

NativeRuntime::NativeRuntime(unsigned)
    : IRRuntime(RuntimeKind::RK_Native), _compiler(std::make_unique<CompilerT>()), _delayed_compilation(false) {
  llvm::sys::getHostCPUFeatures(_host_features);
}

NativeRuntime::~NativeRuntime() {}

void NativeRuntime::link(std::unique_ptr<llvm::Module> M) {

  __verbose("linking");

  _rawM = std::move(M);

  _compiler->prepareModule(*_rawM);

  _M = CloneModule(_rawM.get());
  _M->setDataLayout(_rawM->getDataLayoutStr());

  auto reflect = _M->getFunction("__pacxx_reflect");
  if (!reflect || reflect->getNumUses() == 0) {
    compileAndLink();
  } else {
    _CPUMod = _M.get();
    __verbose("Module contains unresolved calls to __pacxx_reflect. Linking "
                  "delayed!");
    _delayed_compilation = true;
  }
}

void NativeRuntime::compileAndLink() {
  SCOPED_TIMING { _CPUMod = _compiler->compile(_M); };
  _rawM->setDataLayout(_CPUMod->getDataLayout());
  _rawM->setTargetTriple(_CPUMod->getTargetTriple());
  _delayed_compilation = false;
}

Kernel &NativeRuntime::getKernel(const std::string &name) {
  auto It = std::find_if(_kernels.begin(), _kernels.end(),
                         [&](const auto &p) { return name == p.first; });
  if (It == _kernels.end()) {
    void *fptr = nullptr;
    if (!_delayed_compilation) {
      fptr = _compiler->getKernelFptr(_CPUMod, name);
      if (!fptr)
        throw common::generic_exception("Kernel function not found in module!");
    }

    _kernels[name].reset(new NativeKernel(*this, fptr, name));

    return *_kernels[name];
  } else {
    return *It->second;
  }
}

size_t NativeRuntime::getPreferedMemoryAlignment() {
  return _CPUMod->getDataLayout().getPointerABIAlignment();
}

RawDeviceBuffer *NativeRuntime::allocateRawMemory(size_t bytes, MemAllocMode) {
  NativeRawDeviceBuffer rawBuffer;

  rawBuffer.allocate(bytes, 8 * getPreferedVectorSize(1));
  auto wrapped = new NativeDeviceBuffer<char>(std::move(rawBuffer));
  _memory.push_back(std::unique_ptr<DeviceBufferBase>(
      static_cast<DeviceBufferBase *>(wrapped)));
  return wrapped->getRawBuffer();
}

void NativeRuntime::deleteRawMemory(RawDeviceBuffer *ptr) {
  auto It = std::find_if(_memory.begin(), _memory.end(), [&](const auto &uptr) {
    return static_cast<NativeDeviceBuffer<char> *>(uptr.get())
        ->getRawBuffer() == ptr;
  });
  if (It != _memory.end())
    _memory.erase(It);
  else
    __error("ptr to delete not found");
}

void NativeRuntime::requestIRTransformation(Kernel &K) {
  if (_msp_engine.isDisabled())
    return;

  _M = CloneModule(_rawM.get());
  _M->setDataLayout(_rawM->getDataLayoutStr());
  _msp_engine.transformModule(*_M, K);

  compileAndLink();

  void *fptr = nullptr;
  fptr = _compiler->getKernelFptr(_CPUMod, K.getName().c_str());
  static_cast<NativeKernel &>(K).overrideFptr(fptr);
}

void NativeRuntime::synchronize() {};

llvm::legacy::PassManager &NativeRuntime::getPassManager() {
  return _compiler->getPassManager();
};

size_t NativeRuntime::getPreferedVectorSize(size_t dtype_size) {

  for (auto &p : _host_features) {
    if (p.second)
      __verbose(p.first().str());
  }

  if (_host_features["avx-512"])
    return 64 / dtype_size;
  if (_host_features["avx"] || _host_features["avx2"])
    return 32 / dtype_size;
  if (_host_features["sse2"] || _host_features["altivec"])
    return 16 / dtype_size;
  if (_host_features["mmx"])
    return 8 / dtype_size;

  return 1;
}

size_t NativeRuntime::getConcurrentCores() {
  auto n = std::thread::hardware_concurrency();
  if (n == 0)
    return 1;
  return n;
}

bool NativeRuntime::supportsUnifiedAddressing(){
  return true;
}


}
}
