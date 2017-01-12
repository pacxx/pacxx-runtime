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
#include <llvm/ADT/Statistic.h>
#include <llvm/Target/TargetLowering.h>

#include <detail/common/Exceptions.h>
#include <detail/cuda/PTXBackend.h>
#include <llvm/Transforms/PACXXTransforms.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Vectorize.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <detail/common/Timing.h>

using namespace llvm;

namespace pacxx {
  namespace v2 {
    PTXBackend::PTXBackend()
        : _target(nullptr), _cpu("sm_20"), _features("+ptx40"), _pmInitialized(false), _ptxOS(_ptxString) {}

    void PTXBackend::initialize(unsigned CC) {
      _cpu = "sm_" + std::to_string(CC);
      PassRegistry* Registry = PassRegistry::getPassRegistry();
      initializeCore(*Registry);
      initializeCodeGen(*Registry);
      initializeLoopStrengthReducePass(*Registry);
      initializeLowerIntrinsicsPass(*Registry);
      initializeUnreachableMachineBlockElimPass(*Registry);

      _options.LessPreciseFPMADOption = false;
      _options.UnsafeFPMath = false;
      _options.NoInfsFPMath = false;
      _options.NoNaNsFPMath = false;
      _options.HonorSignDependentRoundingFPMathOption = false;
      _options.AllowFPOpFusion = FPOpFusion::Fast;

    }

    std::string PTXBackend::compile(llvm::Module& M) {
      Triple TheTriple = Triple(M.getTargetTriple());
      std::string Error;

      if (!_target)
        _target = TargetRegistry::lookupTarget("nvptx64", TheTriple, Error);
      if (!_target) {
        throw common::generic_exception(Error);
      }

      _ptxString.clear();
      EnableStatistics();
      if (!_pmInitialized) {
        TargetLibraryInfoImpl TLII(Triple(M.getTargetTriple()));
        _PM.add(new TargetLibraryInfoWrapperPass(TLII));

        _PM.add(createReassociatePass());

        _PM.add(createConstantPropagationPass());
        _PM.add(createSCCPPass());
        _PM.add(createConstantHoistingPass());
        _PM.add(createCorrelatedValuePropagationPass());
        _PM.add(createInstructionCombiningPass());
        _PM.add(createLICMPass());

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

	auto RM = Optional<Reloc::Model>();
        _machine.reset(_target->createTargetMachine(
            TheTriple.getTriple(), _cpu, _features, _options, RM,
            CodeModel::Default, CodeGenOpt::None));

        if (_machine->addPassesToEmitFile(_PM, _ptxOS, TargetMachine::CGFT_AssemblyFile,
                                          false)) {
          throw common::generic_exception(
              "target does not support generation of this file type!\n");
        }

        _pmInitialized = true;
      }

      _PM.run(M);
      return _ptxString.str().str();
    }

    llvm::legacy::PassManager& PTXBackend::getPassManager() { return _PM; }
  }
}
