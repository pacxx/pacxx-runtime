//
// Created by mhaidl on 14/06/16.
//

#include "pacxx/detail/native/NativeBackend.h"
#include "pacxx/Executor.h"
#include "pacxx/detail/common/Exceptions.h"
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/LoopPass.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/GVMaterializer.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/LinkAllPasses.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/PACXXTransforms.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Vectorize.h>
#include "llvm/CodeGen/MachineModuleInfo.h"

namespace {
const std::string native_loop_ir(R"(
define void @foo(i32 %__maxx, i32 %__maxy, i32 %__maxz) {
    %1 = alloca i32, align 4
    %2 = alloca i32, align 4
    %3 = alloca i32, align 4
    %__z = alloca i32, align 4
    %__y = alloca i32, align 4
    %__x = alloca i32, align 4
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
)");
}

using namespace llvm;

// native device binding
extern const char native_binding_start[];
extern int native_binding_size;


namespace pacxx {
namespace v2 {
NativeBackend::NativeBackend() : _pmInitialized(false) {}

NativeBackend::~NativeBackend() {}

void NativeBackend::prepareModule(llvm::Module &M) {

  ModuleLoader loader(M.getContext());
  auto binding = loader.loadInternal(native_binding_start, native_binding_size);

  auto linker = Linker(M);
  linker.linkInModule(std::move(binding), Linker::Flags::None);

  llvm::legacy::PassManager PM;

  PM.add(createPACXXTargetSelectPass({"CPU", "Generic"}));
  PM.add(createPACXXInlinerPass());
  PM.add(createPACXXDeadCodeElimPass());
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

  builder.setErrorStr(&error);

  builder.setUseOrcMCJITReplacement(true);

  builder.setEngineKind(EngineKind::JIT);

  builder.setOptLevel(CodeGenOpt::Aggressive);

  _machine = builder.selectTarget(Triple(sys::getProcessTriple()), "",
                                  sys::getHostCPUName(), getTargetFeatures());

  for (auto &F : TheModule->getFunctionList()) {
    F.addFnAttr("target-cpu", _machine->getTargetCPU().str());
    F.addFnAttr("target-features", _machine->getTargetFeatureString().str());
  }
  builder.setMCJITMemoryManager(std::unique_ptr<RTDyldMemoryManager>(
      static_cast<RTDyldMemoryManager *>(new SectionMemoryManager())));

  _JITEngine = builder.create(_machine);

  if (!_JITEngine) {
    throw new common::generic_exception(error);
  }

  TheModule->setTargetTriple(_JITEngine->getTargetMachine()->getTargetTriple().str());
  TheModule->setDataLayout(_JITEngine->getDataLayout());

  raw_fd_ostream OS("moduleBeforePass.ll", EC, sys::fs::F_None);
  TheModule->print(OS, nullptr);

  applyPasses(*TheModule);

  raw_fd_ostream OS1("moduleAfterPass.ll", EC, sys::fs::F_None);
  TheModule->print(OS1, nullptr);

  __verbose("applied pass");

  _JITEngine->finalizeObject();

  return TheModule;
}

SmallVector<std::string, 10> NativeBackend::getTargetFeatures() {
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
  if (!_pmInitialized) {

    PassManagerBuilder builder;
    builder.OptLevel = 3;

    __verbose(_machine->getTargetFeatureString().str());

    TargetLibraryInfoImpl TLII(Triple(M.getTargetTriple()));
    _PM.add(new TargetLibraryInfoWrapperPass(TLII));
    _PM.add(createTargetTransformInfoWrapperPass(_machine->getTargetIRAnalysis()));
    //_PM.add(createPACXXAddrSpaceTransformPass());
    //_PM.add(createPACXXIdRemoverPass());
    _PM.add(createCFGSimplificationPass());
    _PM.add(createLoopSimplifyPass());
    _PM.add(createLCSSAPass());
    _PM.add(createSPMDVectorizerPass());
    _PM.add(createPACXXNativeBarrierPass());
    _PM.add(createPACXXNativeLinkerPass());
    _PM.add(createPACXXNativeSMPass());
    _PM.add(createVerifierPass());
    builder.populateModulePassManager(_PM);

    _pmInitialized = true;
  }

  /*
  std::error_code EC;
  raw_fd_ostream OS2("assembly.asm", EC, sys::fs::F_None);
  _machine->addPassesToEmitFile(_PM, OS2, TargetMachine::CGFT_AssemblyFile);
   */


  _PM.run(M);
}

legacy::PassManager &NativeBackend::getPassManager() { return _PM; }
}
}
