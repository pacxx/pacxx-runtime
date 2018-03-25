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
#include "pacxx/detail/rocm/transforms/Passes.h"

#include "pacxx/ModuleLoader.h"
#include "pacxx/detail/common/Exceptions.h"
#include "pacxx/detail/rocm/HSACOBackend.h"
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

// amdgcn device binding
extern const char amdgcn_binding_start[];
extern const char amdgcn_binding_end[];

using namespace llvm;

namespace pacxx {
namespace v2 {
HSACOBackend::HSACOBackend()
    : _target(nullptr), _cpu("gfx803"), _features("") {}

HSACOBackend::~HSACOBackend() {}

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
}

std::unique_ptr<llvm::Module> HSACOBackend::prepareModule(llvm::Module &M) {

  ModuleLoader loader(M.getContext());
  auto binding = loader.loadInternal(amdgcn_binding_start,
                                     amdgcn_binding_end - amdgcn_binding_start);
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
  PM.add(createAMDGCNPrepairPass(_gcnArch));
  PM.add(createAddressSpaceTransformPass());
  // PM.add(createLoadMotionPass());
  PM.add(createMSPRemoverPass());

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

  llvm::legacy::PassManager PMR;
  PMR.add(new TargetLibraryInfoWrapperPass(TLII));
  PMR.add(createPACXXCodeGenPrepare());
  PMR.add(createAlwaysInlinerLegacyPass());
  PMR.add(createPACXXCodeGenPrepare());

  PMR.run(M);

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
  llvm::TargetOptions options;
  options.UnsafeFPMath = false;
  options.NoInfsFPMath = false;
  options.NoNaNsFPMath = false;
  options.HonorSignDependentRoundingFPMathOption = false;
  options.AllowFPOpFusion = FPOpFusion::Fast;

  M.setTargetTriple("amdgcn--amdhsa-amdgiz");

  M.setDataLayout("e-p:64:64-p1:64:64-p2:64:64-p3:32:32-p4:32:32-p5:32:32-"
                  "i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:"
                  "256-v512:512-v1024:1024-v2048:2048-n32:64-A5");

  Triple TheTriple = Triple(M.getTargetTriple());
  std::string Error;
  SmallString<128> hsaString;
  llvm::raw_svector_ostream OS(hsaString);
  if (!_target)
    _target = TargetRegistry::lookupTarget("amdgcn", TheTriple, Error);
  if (!_target) {
    throw common::generic_exception(Error);
  }

  llvm::legacy::PassManager PM;
  PassManagerBuilder builder;
  builder.OptLevel = 3;
  //  PM.add(createAMDGCNPrepairPass(_gcnArch));
  builder.populateModulePassManager(PM);
  PM.run(M);

  if (common::GetEnv("PACXX_DUMP_FINAL_IR") != "") {
    M.dump();
  }

  llvm::legacy::PassManager lowerPM;
  _machine.reset(_target->createTargetMachine(
      TheTriple.getTriple(), _cpu, _features, options, Reloc::Model::Static,
      CodeModel::Model::Medium, CodeGenOpt::Aggressive));

  if (_machine->addPassesToEmitFile(lowerPM, OS, TargetMachine::CGFT_ObjectFile,
                                    false)) {
    throw std::logic_error(
        "target does not support generation of this file type!\n");
  }

  lowerPM.run(M);

  auto hsa = hsaString.str().str();
  std::ofstream out(".pacxx.isabin");
  out << hsa;

  std::vector<const char *> args;

  std::string outfile = ".pacxx.hsaco";

  auto result =
      std::system(("ld.lld -shared .pacxx.isabin -o " + outfile).c_str());
  if (result < 0)
    __error("Error: ", strerror(errno));
  else {
    if (WIFEXITED(result)) {
      __verbose("ld.lld returned normally, exit code ", WEXITSTATUS(result));
      switch(WEXITSTATUS(result)){
      case 127:
        throw common::generic_exception("ld.lld not found in PATH!");
      case 0: 
        break; 
      default:
        throw common::generic_exception(
            "Unable to execute ld.lld! Check log for error code!"); 
      }
    } else
      __error("ld.lld exited abnormaly\n");
  }
  return outfile;
}

} // namespace v2
} // namespace pacxx
