//
// Created by mhaidl on 29/05/16.
//

#include <llvm/Support/TargetSelect.h>
#include <llvm/CodeGen/LinkAllAsmWriterComponents.h>
#include <llvm/CodeGen/LinkAllCodegenComponents.h>

#include "detail/common/Log.h"
#include "detail/CoreInitializer.h"

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
    auto& log = common::Log::get();
    InitializeAllTargets();
    InitializeAllTargetMCs();
    InitializeAllAsmPrinters();
    InitializeAllAsmParsers();
    _initialized = true;
    __verbose("Core components initialized!");
  }
}
  CoreInitializer::~CoreInitializer() {
  }

}
}