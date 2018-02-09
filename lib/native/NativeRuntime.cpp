//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "pacxx/detail/native/NativeRuntime.h"
#include "pacxx/detail/native/PAPIProfiler.h"
#include "pacxx/detail/common/Exceptions.h"
#include "pacxx/detail/common/Timing.h"
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/LinkAllPasses.h>

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
    : Runtime(RuntimeKind::RK_Native), _compiler(std::make_unique<CompilerT>()), _delayed_compilation(false) {
      #ifdef PACXX_ENABLE_PAPI
      _profiler.reset(new PAPIProfiler());
      #endif
      if (_profiler){
        _profiler->preinit(nullptr);
        _profiler->postinit(nullptr);
      }
}

NativeRuntime::~NativeRuntime() {
  if (_profiler && _profiler->enabled()) 
    _profiler->report();
}

void NativeRuntime::link(std::unique_ptr<llvm::Module> M) {

  __verbose("linking");

  _rawM = std::move(M);

  auto RM = _compiler->prepareModule(*_rawM);
  initializeMSP(std::move(RM));

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
  return _CPUMod->getDataLayout().getPointerABIAlignment(0);
}

std::unique_ptr<RawDeviceBuffer> NativeRuntime::allocateRawMemory(size_t bytes, MemAllocMode mode) {
  return std::unique_ptr<RawDeviceBuffer>(new NativeRawDeviceBuffer(bytes, getPreferedVectorSizeInBytes(), this));
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

void NativeRuntime::synchronize() {}

size_t NativeRuntime::getPreferedVectorSize(size_t dtype_size) {
	return getPreferedVectorSizeInBytes() / dtype_size;
}

size_t NativeRuntime::getPreferedVectorSizeInBytes() {
	llvm::StringMap<bool> host_features;
	llvm::sys::getHostCPUFeatures(host_features);
  if (host_features["avx512f"])
    return 64;
  if (host_features["avx"] || host_features["avx2"])
    return 32;
  if (host_features["sse2"] || host_features["altivec"])
    return 16;
  if (host_features["mmx"])
    return 8;

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
