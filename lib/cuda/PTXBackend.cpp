//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <string>

#include <llvm/ADT/SmallString.h>
#include <llvm/CodeGen/LinkAllAsmWriterComponents.h>
#include <llvm/CodeGen/LinkAllCodegenComponents.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>

#include "pacxx/detail/common/transforms/Passes.h"
#include "pacxx/detail/cuda/transforms/Passes.h"

#include "pacxx/detail/common/Exceptions.h"
#include "pacxx/detail/cuda/PTXBackend.h"

#include "pacxx/ModuleLoader.h"
#include <fstream>
#include <llvm/Analysis/BasicAliasAnalysis.h>
#include <llvm/Analysis/Passes.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Analysis/TypeBasedAliasAnalysis.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/IPO/FunctionAttrs.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Vectorize.h>

#include "pacxx/detail/common/Timing.h"
#include "pacxx/detail/common/transforms/PACXXTransforms.h"
#include "pacxx/detail/common/transforms/Passes.h"

// nvptx device binding
extern const char nvptx_binding_start[];
extern const char nvptx_binding_end[];

using namespace llvm;

namespace pacxx {
namespace v2 {
PTXBackend::PTXBackend()
    : _target(nullptr), _cpu("sm_20"), _features("+ptx40") {}

PTXBackend::~PTXBackend(){}

void PTXBackend::initialize(unsigned CC) {
  _cpu = "sm_" + std::to_string(CC);
  __verbose("Intializing LLVM components for PTX generation!");
  PassRegistry *Registry = PassRegistry::getPassRegistry();
  initializeCore(*Registry);
  initializeCodeGen(*Registry);
  initializeLoopStrengthReducePass(*Registry);
  initializeLowerIntrinsicsPass(*Registry);
  initializeUnreachableMachineBlockElimPass(*Registry);
}

std::unique_ptr<llvm::Module> PTXBackend::prepareModule(llvm::Module &M) {

  ModuleLoader loader(M.getContext());
  auto binding = loader.loadInternal(nvptx_binding_start,
                                     nvptx_binding_end - nvptx_binding_start);

  M.setDataLayout(binding->getDataLayout());
  M.setTargetTriple(binding->getTargetTriple());

  auto linker = Linker(M);
  linker.linkInModule(std::move(binding), Linker::Flags::None);

  llvm::legacy::PassManager PM;
  TargetLibraryInfoImpl TLII(Triple(M.getTargetTriple()));
  PM.add(new TargetLibraryInfoWrapperPass(TLII));
  PM.add(createPACXXCodeGenPrepare());
  PM.add(createAlwaysInlinerLegacyPass());
  PM.add(createPACXXCodeGenPrepare());
  PM.add(createSROAPass());
  PM.add(createPromoteMemoryToRegisterPass());
  PM.add(createLoopRotatePass());
  PM.add(createCFGSimplificationPass());
  //PM.add(createCodeGenPreparePass());
  PM.add(createPostOrderFunctionAttrsLegacyPass());
  PM.add(createSROAPass());
  PM.add(createEarlyCSEPass());
  PM.add(createLazyValueInfoPass());
  PM.add(createCorrelatedValuePropagationPass());
  PM.add(createReassociatePass());
  PM.add(createLCSSAPass());
  PM.add(createLoopRotatePass());
  PM.add(createStraightLineStrengthReducePass());
  PM.add(createLICMPass());
  PM.add(createLoopUnswitchPass());
  PM.add(createLoopIdiomPass());
  PM.add(createLoopDeletionPass());
  PM.add(createLoopUnrollPass());
  PM.add(createInstructionSimplifierPass());
  PM.add(createLCSSAPass());
  PM.add(createGVNPass());
  PM.add(createBreakCriticalEdgesPass());
  PM.add(createConstantMergePass());

  auto PRP = createMSPGenerationPass();
  PM.add(PRP);
  PM.add(createAlwaysInlinerLegacyPass());

  PM.add(createPACXXCodeGenPrepare());
  PM.add(createScalarizerPass());
  PM.add(createPromoteMemoryToRegisterPass());
  PM.add(createInstructionCombiningPass());
  PM.add(createCFGSimplificationPass());
  PM.add(createAlwaysInlinerLegacyPass());
  PM.add(createPACXXCodeGenPrepare());
  PM.add(createIntrinsicSchedulerPass());
  PM.add(createTargetSelectionPass({"GPU", "Generic"}));
  PM.add(createAddressSpaceTransformPass());
  // PM.add(createLoadMotionPass());
  PM.add(createMSPRemoverPass());
  PM.add(createNVPTXPrepairPass());
  PM.add(createIntrinsicMapperPass());
  PM.add(createMemoryCoalescingPass(false));
  PM.add(createAlwaysInlinerLegacyPass());
  PM.add(createPACXXCodeGenPrepare());
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

  PM.run(M);
  auto RM = reinterpret_cast<MSPGeneration *>(PRP)->getReflectionModule();

  PassManagerBuilder builder;
  builder.OptLevel = 3;
  legacy::PassManager RPM;
  RPM.add(createAlwaysInlinerLegacyPass());
  RPM.add(createMSPCleanupPass());
  builder.populateModulePassManager(RPM);
  RPM.run(*RM);

  return RM;
}

std::string PTXBackend::compile(llvm::Module &M) {
  llvm::TargetOptions options;
  options.UnsafeFPMath = false;
  options.NoInfsFPMath = false;
  options.NoNaNsFPMath = false;
  options.HonorSignDependentRoundingFPMathOption = false;
  options.AllowFPOpFusion = FPOpFusion::Fast;

  Triple TheTriple = Triple(M.getTargetTriple());
  std::string Error;
  SmallString<128> ptxString;
  llvm::raw_svector_ostream _ptxOS(ptxString);
  if (!_target)
    _target = TargetRegistry::lookupTarget("nvptx64", TheTriple, Error);
  if (!_target) {
    throw common::generic_exception(Error);
  }

  llvm::legacy::PassManager PM;
  TargetLibraryInfoImpl TLII(Triple(M.getTargetTriple()));
  PM.add(new TargetLibraryInfoWrapperPass(TLII));
  PM.add(createReassociatePass());
  PM.add(createConstantPropagationPass());
  PM.add(createSCCPPass());
  PM.add(createConstantHoistingPass());
  PM.add(createCorrelatedValuePropagationPass());
  PM.add(createInstructionCombiningPass());
  PM.add(createLICMPass());
  PM.add(createIndVarSimplifyPass());
  PM.add(createLoopRotatePass());
  PM.add(createLoopSimplifyPass());
  PM.add(createLoopInstSimplifyPass());
  PM.add(createLCSSAPass());
  PM.add(createLoopStrengthReducePass());
  PM.add(createLICMPass());
  PM.add(createLoopUnrollPass(2000, 32));
  PM.add(createStraightLineStrengthReducePass());
  PM.add(createCorrelatedValuePropagationPass());
  PM.add(createConstantPropagationPass());
  PM.add(createInstructionCombiningPass());
  PM.add(createCFGSimplificationPass());
  PM.add(createInstructionCombiningPass());
  // PM.add(createPACXXStaticEvalPass());
  PM.add(createMemoryCoalescingPass(true));
  PM.add(createDeadInstEliminationPass());
  PM.add(createLoopLoadEliminationPass());
  // PM.add(createLoopIdiomPass());
  PM.add(createLICMPass());
  // PM.add(createEarlyCSEPass());

  if (common::GetEnv("PACXX_PTX_BACKEND_O3") != "") {
    PassManagerBuilder builder;
    builder.OptLevel = 3;
    builder.populateModulePassManager(PM);
  }

  _machine.reset(_target->createTargetMachine(
      TheTriple.getTriple(), _cpu, _features, options, Reloc::Model::Static,
      CodeModel::Model::Medium, CodeGenOpt::None));

  if (_machine->addPassesToEmitFile(PM, _ptxOS,
                                    TargetMachine::CGFT_AssemblyFile, false)) {
    throw common::generic_exception(
        "target does not support generation of this file type!\n");
  }

  PM.run(M);
  if (common::GetEnv("PACXX_DUMP_FINAL_IR") != "") {
    M.dump();
  }

  auto ptx = ptxString.str().str();
  if (common::GetEnv("PACXX_DUMP_ASM") != "") {
    std::ofstream out("dump.ptx");
    out << ptx;
  }
  return ptx;
}

} // namespace v2
} // namespace pacxx
