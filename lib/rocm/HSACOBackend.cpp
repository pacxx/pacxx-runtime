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
#include <llvm/Target/TargetLowering.h>

#include "pacxx/detail/common/transforms/Passes.h"
#include "pacxx/detail/rocm/transforms/Passes.h"

#include "pacxx/detail/common/Exceptions.h"
#include "pacxx/detail/rocm/HSACOBackend.h"

// FIXME: this looks awkward
#include "pacxx/../../../lld/include/lld/Common/Driver.h"

#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Vectorize.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Analysis/BasicAliasAnalysis.h>
#include <llvm/Analysis/TypeBasedAliasAnalysis.h>
#include <llvm/Transforms/IPO/FunctionAttrs.h>
#include <llvm/Analysis/Passes.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/IPO.h>
#include <fstream>
#include "pacxx/ModuleLoader.h"

#include "pacxx/detail/common/transforms/PACXXTransforms.h"
#include "pacxx/detail/common/transforms/Passes.h"
#include "pacxx/detail/common/Timing.h"

// amdgcn device binding
extern const char amdgcn_binding_start[];
extern const char amdgcn_binding_end[];

using namespace llvm;

namespace pacxx {
namespace v2 {
HSACOBackend::HSACOBackend()
    : _target(nullptr), _cpu("gfx803"), _features(""){}

void HSACOBackend::initialize(unsigned gfx) {
  _gcnArch = gfx;
  _cpu = "gfx" + std::to_string(gfx);
  __verbose("Intializing LLVM components for HSACO generation!");
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

std::unique_ptr<llvm::Module> HSACOBackend::prepareModule(llvm::Module &M) {

  ModuleLoader loader(M.getContext());
  auto binding = loader.loadInternal(amdgcn_binding_start, amdgcn_binding_end - amdgcn_binding_start);
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
  PM.add(createCodeGenPreparePass());
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
 // PM.add(createAddressSpaceTransformPass());
 // PM.add(createLoadMotionPass());
  PM.add(createMSPRemoverPass());

  PM.add(createAMDGCNPrepairPass(_gcnArch));

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

std::string HSACOBackend::compile(llvm::Module &M) {
  Triple TheTriple = Triple(M.getTargetTriple());
  std::string Error;
  SmallString<128> ptxString;
  llvm::raw_svector_ostream _ptxOS(ptxString);
  if (!_target)
    _target = TargetRegistry::lookupTarget("amdgcn", TheTriple, Error);
  if (!_target) {
    throw common::generic_exception(Error);
  }

 llvm::legacy::PassManager PM;
  TargetLibraryInfoImpl TLII(Triple(M.getTargetTriple()));
  PassManagerBuilder builder;
  builder.OptLevel = 3;
  builder.populateModulePassManager(PM);

  _machine.reset(_target->createTargetMachine(
      TheTriple.getTriple(), _cpu, _features, _options, Reloc::Model::Static,
      CodeModel::Model::Medium, CodeGenOpt::Aggressive));

  if (_machine->addPassesToEmitFile(
      PM, _ptxOS, TargetMachine::CGFT_ObjectFile, false)) {
    throw std::logic_error(
        "target does not support generation of this file type!\n");
  }

  PM.run(M);

  auto ptx = ptxString.str().str();
    std::ofstream out(".pacxx.isabin");
    out << ptx;

  std::vector<const char *> args; 

  std::string outfile = ".pacxx.hsaco";

  args.push_back("ld.ldd");
  args.push_back("-shared");
  args.push_back(".pacxx.isabin");
  args.push_back("-o");
  args.push_back(outfile.c_str());
 
  lld::elf::link(args, false);

  return outfile;
}

}
}
