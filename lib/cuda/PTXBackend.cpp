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

#include "pacxx/detail/common/Exceptions.h"
#include "pacxx/detail/cuda/PTXBackend.h"
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Transforms/PACXXTransforms.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Vectorize.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include "pacxx/ModuleLoader.h"

#include "pacxx/detail/common/Timing.h"

// nvptx device binding
extern const char nvptx_binding_start[];
extern const char nvptx_binding_end[];

using namespace llvm;

namespace pacxx {
namespace v2 {
PTXBackend::PTXBackend()
    : _target(nullptr), _cpu("sm_20"), _features("+ptx40"),
      _pmInitialized(false) {}

void PTXBackend::initialize(unsigned CC) {
  _cpu = "sm_" + std::to_string(CC);
  __verbose("Intializing LLVM components for PTX generation!");
  PassRegistry *Registry = PassRegistry::getPassRegistry();
  initializeCore(*Registry);
  initializeCodeGen(*Registry);
  initializeLoopStrengthReducePass(*Registry);
  initializeLowerIntrinsicsPass(*Registry);
  initializeUnreachableMachineBlockElimPass(*Registry);

  _options.UnsafeFPMath = false;
  _options.NoInfsFPMath = false;
  _options.NoNaNsFPMath = false;
  _options.HonorSignDependentRoundingFPMathOption = false;
  _options.AllowFPOpFusion = FPOpFusion::Fast;
}

void PTXBackend::prepareModule(llvm::Module &M) {

  ModuleLoader loader(M.getContext());
  auto binding = loader.loadInternal(nvptx_binding_start, nvptx_binding_end - nvptx_binding_start);

  M.setDataLayout(binding->getDataLayout());
  M.setTargetTriple(binding->getTargetTriple());

  auto linker = Linker(M);
  linker.linkInModule(std::move(binding), Linker::Flags::None);

  llvm::legacy::PassManager PM;

  PM.add(createPACXXTargetSelectPass({"GPU", "Generic"}));
  PM.add(createPACXXSpirPass());
  PM.add(createPACXXReflectionRemoverPass());
  PM.add(createPACXXNvvmPass());

  PM.add(createPACXXNvvmRegPass(false));
  PM.add(createPACXXInlinerPass());
  PM.add(createPACXXDeadCodeElimPass());
  PM.add(createCFGSimplificationPass());
  PM.add(createInferAddressSpacesPass());
  PM.add(createSROAPass());
  PM.add(createPromoteMemoryToRegisterPass());
  PM.add(createDeadStoreEliminationPass());
  PM.add(createInstructionCombiningPass());
  PM.add(createCFGSimplificationPass());
  PM.add(createSROAPass());
  PM.add(createPromoteMemoryToRegisterPass());
  PM.add(createInstructionCombiningPass());
  PM.add(createInferAddressSpacesPass());

  PM.run(M);
}

std::string PTXBackend::compile(llvm::Module &M) {
  Triple TheTriple = Triple(M.getTargetTriple());
  std::string Error;
  llvm::raw_svector_ostream _ptxOS(_ptxString);
  if (!_target)
    _target = TargetRegistry::lookupTarget("nvptx64", TheTriple, Error);
  if (!_target) {
    throw common::generic_exception(Error);
  }


  _ptxString.clear();
  if (!_pmInitialized) {
    TargetLibraryInfoImpl TLII(Triple(M.getTargetTriple()));
    _PM.add(new TargetLibraryInfoWrapperPass(TLII));
    _PM.add(createReassociatePass());
    _PM.add(createInferAddressSpacesPass());
    _PM.add(createConstantPropagationPass());
    _PM.add(createSCCPPass());
    _PM.add(createConstantHoistingPass());
    _PM.add(createCorrelatedValuePropagationPass());
    _PM.add(createInstructionCombiningPass());
    _PM.add(createLICMPass());
    _PM.add(createInferAddressSpacesPass());
    _PM.add(createIndVarSimplifyPass());
    _PM.add(createLoopRotatePass());
    _PM.add(createLoopSimplifyPass());
    _PM.add(createLoopInstSimplifyPass());
    _PM.add(createLCSSAPass());
    _PM.add(createLoopStrengthReducePass());
    _PM.add(createLICMPass());
    _PM.add(createLoopUnrollPass(2000, 32));
    _PM.add(createStraightLineStrengthReducePass());
    _PM.add(createCorrelatedValuePropagationPass());
    _PM.add(createConstantPropagationPass());
    _PM.add(createInstructionCombiningPass());
    _PM.add(createCFGSimplificationPass());
    _PM.add(createInstructionCombiningPass());
    _PM.add(createPACXXStaticEvalPass());
    _PM.add(createPACXXNvvmRegPass(true));

    if (common::GetEnv("PACXX_PTX_BACKEND_O3") != ""){
      PassManagerBuilder builder;
      builder.OptLevel = 3;
      builder.populateModulePassManager(_PM);
    }

    _machine.reset(_target->createTargetMachine(
        TheTriple.getTriple(), _cpu, _features, _options, Reloc::Model::Static,
        CodeModel::Default, CodeGenOpt::None));

    if (_machine->addPassesToEmitFile(
            _PM, _ptxOS, TargetMachine::CGFT_AssemblyFile, false)) {
      throw common::generic_exception(
          "target does not support generation of this file type!\n");
    }

    _pmInitialized = true;
  }

  _PM.run(M);
  return _ptxString.str().str();
}

llvm::legacy::PassManager &PTXBackend::getPassManager() { return _PM; }
}
}
