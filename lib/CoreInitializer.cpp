//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <llvm/CodeGen/LinkAllAsmWriterComponents.h>
#include <llvm/CodeGen/LinkAllCodegenComponents.h>
#include <llvm/Support/TargetSelect.h>

#include "pacxx/detail/CoreInitializer.h"
#include "pacxx/detail/common/Log.h"
#include "pacxx/detail/common/TearDown.h"

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
    std::atexit(v2::pacxxTearDown);
    __verbose("Core components initialized!");
  }
}
CoreInitializer::~CoreInitializer() {}
}
}