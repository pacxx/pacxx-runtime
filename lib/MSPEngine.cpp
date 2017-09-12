//
// Created by mhaidl on 30/06/16.
//
#include "pacxx/detail/msp/MSPEngine.h"
#include "pacxx/detail/common/Exceptions.h"
#include "pacxx/detail/common/LLVMHelper.h"
#include "pacxx/detail/common/Timing.h"
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/Constants.h>

#include "pacxx/detail/common/transforms/ModuleHelper.h"

using namespace llvm;

namespace pacxx {
namespace v2 {

MSPEngine::MSPEngine() :  _disabled(true), _engine(nullptr) {}

void MSPEngine::initialize(std::unique_ptr<llvm::Module> M) {
  _mspModule = M.get();
  _stubs = pacxx::getTagedFunctionsWithTag(M.get(), "pacxx.reflection", "stub");
  if (!_stubs.empty()) {
    std::string ErrStr;

    EngineBuilder builder{std::move(M)};

    builder.setErrorStr(&ErrStr);
    builder.setUseOrcMCJITReplacement(true);
    builder.setEngineKind(EngineKind::JIT);

    RTDyldMemoryManager *RTDyldMM = new SectionMemoryManager();

    builder.setMCJITMemoryManager(
        std::unique_ptr<RTDyldMemoryManager>(RTDyldMM));
    _engine = builder.create();
    if (!_engine) {
      __error(ErrStr);
      throw new pacxx::common::generic_exception(ErrStr);
    }

    _engine->finalizeObject();

    __verbose("MSP Engine initialized: ", _stubs.size(), " stubs");
    _disabled = false;
  } else {
    __debug("MSP Engine disabled!");
  }
}

size_t MSPEngine::getArgBufferSize(const llvm::Function &KF, Kernel &kernel) {
  size_t hostArgBufferSize = 0;
  if (!kernel.requireStaging())
    return hostArgBufferSize;
  __verbose("reading ArgBufferSize for function: ", KF.getName().str());
  auto &M = *KF.getParent();
  if (auto RF = M.getFunction("__pacxx_reflect")) {
    for (auto U : RF->users()) {
      if (CallInst *CI = dyn_cast<CallInst>(U)) {
        if (CI->getParent()->getParent() == &KF) {
          if (MDNode *MD = CI->getMetadata("pacxx.reflect.stage")) {
            auto *ci32 = dyn_cast<ConstantInt>(
                dyn_cast<ValueAsMetadata>(MD->getOperand(0).get())->getValue());
            auto cstage = (unsigned int) *ci32->getValue().getRawData();
            auto FName =
                std::string("__pacxx_reflection_stub") + std::to_string(cstage);

            if (auto F = _engine->FindFunctionNamed(FName.c_str())) {
              auto MD = F->getMetadata("pacxx.reflection.argBufferSize");
              hostArgBufferSize =
                  *(dyn_cast<ConstantInt>(
                      dyn_cast<ValueAsMetadata>(MD->getOperand(0).get())
                          ->getValue())
                      ->getValue()
                      .getRawData());
            }
          }
        }
      }
    }
  }
  return hostArgBufferSize;
}

void MSPEngine::evaluate(const llvm::Function &KF, Kernel &kernel) {
  if (!kernel.requireStaging())
    return;
  __verbose("staging function: ", KF.getName().str());
  bool kernelHasStagedFunction = false;
  auto &M = *KF.getParent();

  if (auto RF = M.getFunction("__pacxx_reflect")) {
    for (auto U : RF->users()) {
      if (CallInst *CI = dyn_cast<CallInst>(U)) {
        int64_t value = 0;
        bool inScope = false;
        if (CI->getParent()->getParent() == &KF) {
          if (MDNode *MD = CI->getMetadata("pacxx.reflect.stage")) {
            auto *ci32 = dyn_cast<ConstantInt>(
                dyn_cast<ValueAsMetadata>(MD->getOperand(0).get())->getValue());
            auto cstage = (unsigned int) *ci32->getValue().getRawData();
            auto FName =
                std::string("__pacxx_reflection_stub") + std::to_string(cstage);

            if (auto F = _mspModule->getFunction(FName)) {
              void *rFP = _engine->getPointerToFunction(F);
              auto FP = reinterpret_cast<int64_t (*)(const void *)>(rFP);
              auto ptr = kernel.getLambdaPtr();
              value = FP(&ptr);

              kernelHasStagedFunction = true;
              inScope = true;
            }
            __verbose("staging: ", FName, "  - result is ", value, " ", (inScope ? "" : "evaluation not in scope"));

            if (auto *ci2 = dyn_cast<ConstantInt>(CI->getOperand(0))) {
              kernel.setStagedValue(*(ci2->getValue().getRawData()), value,
                                    inScope);
            }
          }
        }
      }
    }
  }
  if (!kernelHasStagedFunction) {
    __verbose("disabling staging for: ", KF.getName().str());
    kernel.disableStaging();
  }
}

void MSPEngine::transformModule(llvm::Module &M, Kernel &K) {
  common::CallFinder finder;
  auto &staged_values = K.getStagedValues();
  if (Function *RF = M.getFunction("__pacxx_reflect")) {
    for (auto U : RF->users()) {
      if (CallInst *CI = dyn_cast<CallInst>(U)) {
        if (auto *ci2 = dyn_cast<ConstantInt>(CI->getOperand(0))) {
          int rep = *(ci2->getValue().getRawData());
          int64_t value = 0;
          for (auto p : staged_values) {
            if (p.first >= 0 && p.first == rep) {
              value = p.second;
            }
          }
          CI->replaceAllUsesWith(ConstantInt::get(CI->getType(), value));
          CI->eraseFromParent();
        }
      }
    }
  }

  return;

  std::vector<int64_t> conf(6);
  for (auto p : staged_values) {
    if (p.first < 0) {
      conf[(p.first + 1) * -1] = p.second;
    }
  }
  Function *KF = M.getFunction(K.getName());

  auto kernelMD =
      M.getOrInsertNamedMetadata(std::string("pacxx.kernel.") + K.getName());
  std::vector<Metadata *> MDArgs;

  int op = -1;
  for (unsigned i = 0; i < kernelMD->getNumOperands(); ++i) {
    auto MD = kernelMD->getOperand(i);
    if (auto ID = dyn_cast<MDString>(MD->getOperand(0))) {
      if (ID->getString().equals("launch config")) {
        op = i;
        break;
      }
    }
  }

  MDArgs.push_back(MDString::get(M.getContext(), "launch config"));
  for (auto v : conf) {
    MDArgs.push_back(llvm::ConstantAsMetadata::get(
        ConstantInt::get(IntegerType::getInt32Ty(M.getContext()), v)));
  }
  if (op < 0)
    kernelMD->addOperand(MDNode::get(M.getContext(), MDArgs));
  else
    kernelMD->setOperand(op, MDNode::get(M.getContext(), MDArgs));

  for (size_t i = 0; i < conf.size(); ++i) {
    Intrinsic::ID iid;
    Function *F = nullptr;
    switch (i) {
    case 0:iid = Intrinsic::nvvm_read_ptx_sreg_ntid_x;
      F = M.getFunction("_Z14get_local_sizej");
      break;
    case 1:iid = Intrinsic::nvvm_read_ptx_sreg_ntid_y;
      F = M.getFunction("_Z14get_local_sizej");
      break;
    case 2:iid = Intrinsic::nvvm_read_ptx_sreg_ntid_z;
      F = M.getFunction("_Z14get_local_sizej");
      break;
    case 3:iid = Intrinsic::nvvm_read_ptx_sreg_nctaid_x;
      F = M.getFunction("_Z14get_num_groupsj");
      break;
    case 4:iid = Intrinsic::nvvm_read_ptx_sreg_nctaid_y;
      F = M.getFunction("_Z14get_num_groupsj");
      break;
    case 5:iid = Intrinsic::nvvm_read_ptx_sreg_nctaid_z;
      F = M.getFunction("_Z14get_num_groupsj");
      break;
    }
    unsigned v = i;

    if (v >= 3)
      v -= 3;

    finder.setIntrinsicID(iid);
    finder.setOpenCLFunction(F, v);

    finder.visit(KF);

    const auto &calls = finder.getFoundCalls();
    for (const auto &CI : calls) {
      __verbose("replacing ntid/nctaid call with ", conf[i]);
      auto IF = CI->getCalledFunction();
      CI->replaceAllUsesWith(ConstantInt::get(CI->getType(), conf[i]));
      CI->eraseFromParent();
      if (IF->hasNUses(0))
        IF->eraseFromParent();
    }

    finder.reset();
  }
}

bool MSPEngine::isDisabled() { return _disabled; }
}
}
