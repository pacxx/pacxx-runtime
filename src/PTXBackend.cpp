//
// Created by mhaidl on 29/05/16.
//
#include <string>

#include <llvm/ADT/SmallString.h>
#include <llvm/CodeGen/LinkAllAsmWriterComponents.h>
#include <llvm/CodeGen/LinkAllCodegenComponents.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Target/TargetLowering.h>
#include <llvm/Target/TargetMachine.h>

#include "detail/Exceptions.h"
#include "detail/PTXBackend.h"

using namespace llvm;

namespace pacxx {
namespace v2 {
PTXBackend::PTXBackend()
    : _target(nullptr), _cpu("sm_35"), _features("+ptx40") {}

void PTXBackend::initialize() {
  PassRegistry *Registry = PassRegistry::getPassRegistry();
  initializeCore(*Registry);
  initializeCodeGen(*Registry);
  initializeLoopStrengthReducePass(*Registry);
  initializeLowerIntrinsicsPass(*Registry);
  initializeUnreachableBlockElimPass(*Registry);

  _options.LessPreciseFPMADOption = false;
  _options.UnsafeFPMath = false;
  _options.NoInfsFPMath = false;
  _options.NoNaNsFPMath = false;
  _options.HonorSignDependentRoundingFPMathOption = false;
  _options.AllowFPOpFusion = FPOpFusion::Fast;
}

std::string PTXBackend::compile(std::unique_ptr<llvm::Module> &M) {
  Triple TheTriple = Triple(M->getTargetTriple());
  std::string Error;

  if (!_target)
    _target = TargetRegistry::lookupTarget("nvptx64", TheTriple, Error);
  if (!_target) {
    throw common::generic_exception(Error);
  }

  _machine.reset(_target->createTargetMachine(
      TheTriple.getTriple(), _cpu, _features, _options, Reloc::Default,
      CodeModel::Default, CodeGenOpt::None));

  SmallString<128> ptxString;
 // raw_svector_ostream ptxOS(ptxString);
 // ptxOS.SetUnbuffered();
  std::error_code EC;
  sys::fs::OpenFlags flags;
  raw_fd_ostream ptxOS("test.ptx", EC, flags);

  legacy::PassManager PM;

  if (_machine->addPassesToEmitFile(PM, ptxOS, TargetMachine::CGFT_AssemblyFile,
                                    false)) {
    throw common::generic_exception(
        "target does not support generation of this file type!\n");
  }

  PM.run(*M);

  std::stringstream ss;
  for (auto x : ptxString)
    ss << x;

  __message(ptxString.str().str());

  return ss.str();
}
}
}