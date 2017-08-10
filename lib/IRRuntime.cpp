//
// Created by m_haid02 on 28.04.17.
//

#include "pacxx/detail/Kernel.h"
#include "pacxx/detail/IRRuntime.h"

namespace pacxx {
namespace v2 {

IRRuntime::IRRuntime(RuntimeKind kind) : _kind(kind), _msp_engine() {}

void IRRuntime::initializeMSP(std::unique_ptr<llvm::Module> M) {
  if (!_msp_engine.isDisabled())
    return;
  _msp_engine.initialize(std::move(M));
}

void IRRuntime::evaluateStagedFunctions(Kernel &K) {
  if (K.requireStaging()) {
    if (_msp_engine.isDisabled())
      return;
    _msp_engine.evaluate(*_rawM->getFunction(K.getName()), K);
  }
}

const llvm::Module &IRRuntime::getModule() { return *_rawM; }
}
}