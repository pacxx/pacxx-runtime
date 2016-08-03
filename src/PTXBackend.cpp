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

#include <detail/common/Exceptions.h>
#include <detail/cuda/PTXBackend.h>
#include <llvm/Transforms/PACXXTransforms.h>

using namespace llvm;

namespace pacxx {
  namespace v2 {
    PTXBackend::PTXBackend()
        : _target(nullptr), _cpu("sm_20"), _features("+ptx40"), _pmInitialized(false), _ptxOS(_ptxString) {}

    void PTXBackend::initialize(unsigned CC) {
      _cpu = "sm_" + std::to_string(CC);
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

    std::string PTXBackend::compile(llvm::Module& M) {
      Triple TheTriple = Triple(M.getTargetTriple());
      std::string Error;

      if (!_target)
        _target = TargetRegistry::lookupTarget("nvptx64", TheTriple, Error);
      if (!_target) {
        throw common::generic_exception(Error);
      }

      _ptxString.clear();

      if (!_pmInitialized) {
        _machine.reset(_target->createTargetMachine(
            TheTriple.getTriple(), _cpu, _features, _options, Reloc::Default,
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