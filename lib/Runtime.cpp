//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "pacxx/detail/Kernel.h"
#include "pacxx/detail/Runtime.h"
#include <llvm/IR/Module.h>
namespace pacxx {
namespace v2 {

Runtime::Runtime(RuntimeKind kind) : _kind(kind), _msp_engine() {
  _profiler.reset(new Profiler());
}

Runtime::~Runtime(){}

void Runtime::initializeMSP(std::unique_ptr<llvm::Module> M) {
  if (!_msp_engine.isDisabled())
    return;
  _msp_engine.initialize(std::move(M));
}

void Runtime::evaluateStagedFunctions(Kernel &K) {
  if (K.requireStaging()) {
    if (_msp_engine.isDisabled())
      return;
    _msp_engine.evaluate(*_rawM->getFunction(K.getName()), K);
  }
}

void Runtime::restoreMemory() {
  if (_profiler->enabled())
  {
    for (std::unique_ptr<DeviceBufferBase<void>>& entry : _memory) {
      entry.get()->restore();
    }
  }
}

const llvm::Module &Runtime::getModule() { return *_rawM; }
}
}
