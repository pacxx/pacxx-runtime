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
#include "pacxx/detail/native/NativeEvent.h"
#include <memory>
#include <llvm/IR/Module.h>

namespace pacxx {
namespace v2 {

Runtime::Runtime(RuntimeKind kind) : _kind(kind), _msp_engine() {
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

void Runtime::enshadowMemory() {
  if (_profiler && _profiler->enabled())
  {
    for (std::unique_ptr<DeviceBufferBase<void>>& entry : _memory) {
      if (entry)
        entry->enshadow();
    }
  }
}

void Runtime::restoreMemory() {
  if (_profiler && _profiler->enabled())
  {
    for (std::unique_ptr<DeviceBufferBase<void>>& entry : _memory) {
      if (entry)
        entry->restore();
    }
  }
}

const llvm::Module &Runtime::getModule() { return *_rawM; }

std::unique_ptr<Event> Runtime::createEvent(){
  __verbose("Runtime does not override Event creation using NativeEvent with host side timers!");
  return std::unique_ptr<Event>(new NativeEvent());
}

}
}
