//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "pacxx/detail/native/NativeBackend.h"
#include "pacxx/ModuleLoader.h"
#include "pacxx/detail/common/Common.h"
#include "pacxx/detail/common/Exceptions.h"
#include "pacxx/detail/common/transforms/Passes.h"
#include "pacxx/detail/native/transforms/Passes.h"
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/LoopPass.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/CodeGen/MachineModuleInfo.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/GVMaterializer.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/LinkAllPasses.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Vectorize.h>

#include "pacxx/detail/common/transforms/PACXXTransforms.h"
#include "pacxx/detail/common/transforms/Passes.h"
namespace {
const std::string native_loop_ir(R"(
define void @__pacxx_block(i32 %__maxx, i32 %__maxy, i32 %__maxz) {
    %1 = alloca i32, align 4
    %2 = alloca i32, align 4
    %3 = alloca i32, align 4
    %__z = alloca i32, align 4, !pacxx_read_tid_z !3
    %__y = alloca i32, align 4, !pacxx_read_tid_y !3
    %__x = alloca i32, align 4, !pacxx_read_tid_x !3
    store i32 %__maxx, i32* %1, align 4
    store i32 %__maxy, i32* %2, align 4
    store i32 %__maxz, i32* %3, align 4
    store i32 0, i32* %__z, align 4
    br label %4

    ; <label>:4                                       ; preds = %27, %0
    %5 = load i32, i32* %__z, align 4
    %6 = load i32, i32* %3, align 4
    %7 = icmp ult i32 %5, %6
    br i1 %7, label %8, label %30

    ; <label>:8                                       ; preds = %4
    store i32 0, i32* %__y, align 4
    br label %9

    ; <label>:9                                       ; preds = %23, %8
    %10 = load i32, i32* %__y, align 4
    %11 = load i32, i32* %2, align 4
    %12 = icmp ult i32 %10, %11
    br i1 %12, label %13, label %26

    ; <label>:13                                      ; preds = %9
    store i32 0, i32* %__x, align 4
    br label %14

    ; <label>:14                                      ; preds = %19, %13
    %15 = load i32, i32* %__x, align 4
    %16 = load i32, i32* %1, align 4
    %17 = icmp ult i32 %15, %16
    br i1 %17, label %18, label %22

    ; <label>:18                                      ; preds = %14
    call void @__dummy_kernel()
    br label %19

    ; <label>:19                                      ; preds = %18
    %20 = load i32, i32* %__x, align 4
    %21 = add i32 %20, 1
    store i32 %21, i32* %__x, align 4
    br label %14, !llvm.loop !1

    ; <label>:22                                      ; preds = %14
    br label %23

    ; <label>:23                                      ; preds = %22
    %24 = load i32, i32* %__y, align 4
    %25 = add i32 %24, 1
    store i32 %25, i32* %__y, align 4
    br label %9

    ; <label>:26                                      ; preds = %9
    br label %27

    ; <label>:27                                      ; preds = %26
    %28 = load i32, i32* %__z, align 4
    %29 = add i32 %28, 1
    store i32 %29, i32* %__z, align 4
    br label %4

    ; <label>:30                                      ; preds = %4
    ret void
}

declare void @__dummy_kernel()

!llvm.ident = !{!0}

!0 = !{!"PACXX"}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.vectorize.enable", i1 false}
!3 = !{null}
)");
}

using namespace llvm;

// native device binding
extern const char native_binding_start[];
extern const char native_binding_end[];

namespace {
static SmallVector<std::string, 10> getTargetFeatures() {
  StringMap<bool> HostFeatures;
  SmallVector<std::string, 10> attr;

  llvm::sys::getHostCPUFeatures(HostFeatures);

  for (StringMap<bool>::const_iterator it = HostFeatures.begin();
       it != HostFeatures.end(); it++) {
    std::string att = it->getValue() ? it->getKey().str()
                                     : std::string("-") + it->getKey().str();
    attr.append(1, att);
  }

  return attr;
}
} // namespace

namespace pacxx {
namespace v2 {
NativeBackend::NativeBackend()
    : _disableVectorizer(false),
      _disableSelectEmitter(false), _disableExpPasses(false) {
  if (common::GetEnv("PACXX_DISABLE_RV") != "")
    _disableVectorizer = true;
  if (common::GetEnv("PACXX_DISABLE_SE") != "")
    _disableSelectEmitter = true;
  if (common::GetEnv("PACXX_DISABLE_EXP_PASS") != "")
    _disableExpPasses = true;
}

NativeBackend::~NativeBackend() {}

std::unique_ptr<llvm::Module> NativeBackend::prepareModule(llvm::Module &M) {

  ModuleLoader loader(M.getContext());
  auto binding = loader.loadInternal(native_binding_start,
                                     native_binding_end - native_binding_start);

  auto linker = Linker(M);
  linker.linkInModule(std::move(binding), Linker::Flags::None);
  
  llvm::legacy::PassManager PM;
  TargetLibraryInfoImpl TLII(Triple(M.getTargetTriple()));
  PM.add(new TargetLibraryInfoWrapperPass(TLII));
  PM.add(createPACXXCodeGenPrepare());
  PM.add(createIntrinsicMapperPass());
  PM.add(createSROAPass());
  PM.add(createPromoteMemoryToRegisterPass());
  PM.add(createLoopRotatePass());
  PM.add(createCFGSimplificationPass());
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

  PM.add(createMSPRemoverPass());
  PM.add(createTargetSelectionPass({"CPU", "Generic"}));
  PM.add(createAlwaysInlinerLegacyPass());
  PM.add(createPACXXCodeGenPrepare());
  PM.add(createCFGSimplificationPass());
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

Module *NativeBackend::compile(std::unique_ptr<Module> &M) {

  std::string error;
  std::error_code EC;

  LLVMInitializeNativeTarget();
  LLVMInitializeNativeAsmPrinter();
  LLVMInitializeNativeAsmParser();

  linkInModule(M);

  Module *TheModule = _composite.get();
  EngineBuilder builder{std::move(_composite)};

  auto triple = Triple(sys::getProcessTriple());

  _machine = builder.selectTarget(Triple(sys::getProcessTriple()), "",
                                  sys::getHostCPUName(), getTargetFeatures());

  TheModule->setTargetTriple(_machine->getTargetTriple().str());
  TheModule->setDataLayout(_machine->createDataLayout());
  for (auto &F : TheModule->getFunctionList()) {
    F.addFnAttr("target-cpu", _machine->getTargetCPU().str());
    F.addFnAttr("target-features", _machine->getTargetFeatureString().str());
  }
  raw_fd_ostream OS("moduleBeforePass.ll", EC, sys::fs::F_None);
  TheModule->print(OS, nullptr);

  applyPasses(*TheModule);

  raw_fd_ostream OS1("moduleAfterPass.ll", EC, sys::fs::F_None);
  TheModule->print(OS1, nullptr);

  __verbose("applied pass");

  builder.setErrorStr(&error);

  builder.setUseOrcMCJITReplacement(false);

  builder.setEngineKind(EngineKind::JIT);

  builder.setOptLevel(CodeGenOpt::Aggressive);

  builder.setMCJITMemoryManager(std::unique_ptr<RTDyldMemoryManager>(
      static_cast<RTDyldMemoryManager *>(new SectionMemoryManager())));

  _JITEngine = builder.create(_machine);

  if (!_JITEngine) {
    throw new common::generic_exception(error);
  }

  _JITEngine->finalizeObject();

  return TheModule;
}

void *NativeBackend::getKernelFptr(Module *module, const std::string name) {
  Function *kernel = module->getFunction("__wrapped__" + name);
  // get the kernel wrapper function from the module
  return _JITEngine->getPointerToFunction(kernel);
}

void NativeBackend::linkInModule(std::unique_ptr<Module> &M) {
  _composite = std::make_unique<Module>("pacxx-link", M->getContext());
  std::unique_ptr<Module> functionModule =
      NativeBackend::createModule(M->getContext(), native_loop_ir);
  auto linker = Linker(*_composite);
  linker.linkInModule(std::move(functionModule), Linker::Flags::None);
  linker.linkInModule(std::move(M), Linker::Flags::None);
  _composite->setTargetTriple(sys::getProcessTriple());
}

std::unique_ptr<Module> NativeBackend::createModule(LLVMContext &Context,
                                                    const std::string IR) {
  SMDiagnostic Err;
  MemoryBufferRef buffer(IR, "loop-buffer");
  std::unique_ptr<Module> Result = parseIR(buffer, Err, Context);
  if (!Result)
    Err.print("createModule", errs());
  return Result;
}

void NativeBackend::applyPasses(Module &M) {
  if (!_machine)
    throw common::generic_exception("Can not get target machine");
  llvm::legacy::PassManager PM;

  PassManagerBuilder builder;
  builder.OptLevel = 3;

  __verbose(_machine->getTargetFeatureString().str());

  TargetLibraryInfoImpl TLII(Triple(M.getTargetTriple()));
  PM.add(new TargetLibraryInfoWrapperPass(TLII));
  PM.add(createTargetTransformInfoWrapperPass(_machine->getTargetIRAnalysis()));
  PM.add(createCFGSimplificationPass());
  PM.add(createLoopSimplifyPass());
  PM.add(createLCSSAPass());

  PM.add(createEarlyCSEPass(true));
  if (!_disableExpPasses) {
    PM.add(createMemoryCoalescingPass(false));
    PM.add(createDeadInstEliminationPass());
  }
  PM.add(createLowerSwitchPass());
  // builder.populateModulePassManager(PM);
  if (!_disableVectorizer) {
    //   PM.add(createPACXXCodeGenPrepare());
    PM.add(createSPMDVectorizerPass());
  }
  PM.add(createAlwaysInlinerLegacyPass());
  if (!_disableSelectEmitter)
    PM.add(createMaskedMemTransformPass());
  PM.add(createIntrinsicSchedulerPass());
  PM.add(createBarrierGenerationPass());
  // PM.add(createVerifierPass());

  PM.add(createKernelLinkerPass());

  PM.add(createSMGenerationPass());

  PM.add(createVerifierPass());
  builder.populateModulePassManager(PM);

  if (common::GetEnv("PACXX_DUMP_ASM") != "") {
    std::error_code EC;
    raw_fd_ostream OS2("dump.asm", EC, sys::fs::F_None);
    _machine->addPassesToEmitFile(PM, OS2, TargetMachine::CGFT_AssemblyFile);
  }

  PM.run(M);

  if (common::GetEnv("PACXX_DUMP_FINAL_IR") != "") {
    M.dump();
  }
}

} // namespace v2
} // namespace pacxx
