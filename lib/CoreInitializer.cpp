//
// Created by mhaidl on 29/05/16.
//

#include <llvm/CodeGen/LinkAllAsmWriterComponents.h>
#include <llvm/CodeGen/LinkAllCodegenComponents.h>
#include <llvm/Support/TargetSelect.h>

#include "pacxx/detail/CoreInitializer.h"
#include "pacxx/detail/common/Log.h"

using namespace llvm;

namespace pacxx {
namespace core {

void CoreInitializer::initialize() {
  static CoreInitializer the_core;
  the_core.initializeCore();
}

CoreInitializer::CoreInitializer() : _initialized(false) {}

void CoreInitializer::initializeCore() {
  if (!_initialized) {
    common::Log::get();
    InitializeAllTargets();
    InitializeAllTargetMCs();
    InitializeAllAsmPrinters();
    InitializeAllAsmParsers();
    _initialized = true;
    __verbose("Core components initialized!");
  }
}
CoreInitializer::~CoreInitializer() {}
}
}