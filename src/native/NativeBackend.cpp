//
// Created by mhaidl on 14/06/16.
//

#include "detail/native/NativeBackend.h"
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Transforms/PACXXTransforms.h>
#include <detail/common/Exceptions.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/Bitcode/ReaderWriter.h>
#include <llvm/Transforms/Scalar.h>
#include "llvm/Transforms/Vectorize.h"

namespace {
  const std::string native_loop_ir(R"(
  define void @foo(i32 %__maxx, i32 %__maxy, i32 %__maxz) #0 {
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

  ; <label>:4                                       ; preds = %33, %0
  %5 = load i32, i32* %__z, align 4
  %6 = load i32, i32* %3, align 4
  %7 = icmp ult i32 %5, %6
  br i1 %7, label %8, label %38

  ; <label>:8                                       ; preds = %4
  store i32 0, i32* %__y, align 4
  br label %9

  ; <label>:9                                       ; preds = %29, %8
  %10 = load i32, i32* %__y, align 4
  %11 = load i32, i32* %2, align 4
  %12 = icmp ult i32 %10, %11
  br i1 %12, label %13, label %35

  ; <label>:13                                      ; preds = %9
  store i32 0, i32* %__x, align 4
  br label %14

  ; <label>:14                                      ; preds = %13
  %15 = load i32, i32* %__x, align 4
  %16 = load i32, i32* %1, align 4
  %17 = icmp ult i32 %15, %16
  br i1 %17, label %18, label %30

  ; <label>:18                                      ; preds = %14
  %19 = load i32, i32* %__x, align 4
  %20 = zext i32 %19 to i64
  %21 = load i32, i32* %__y, align 4
  %22 = zext i32 %21 to i64
  %23 = load i32, i32* %__z, align 4
  %24 = zext i32 %23 to i64
  call void @__dummy_kernel(i64 %20, i64 %22, i64 %24)
  %25 = load i32, i32* %__x, align 4
  %26 = add i32 %25, 1
  store i32 %26, i32* %__x, align 4
  %27 = load i32, i32* %__x, align 4
  %28 = load i32, i32* %1, align 4
  %29 = icmp ult i32 %27, %28
  br i1 %29, label %18, label %30

  ; <label>:30                                      ; preds = %18
  br label %31

  ; <label>:31                                      ; preds = %30
  %32 = load i32, i32* %__y, align 4
  %33 = add i32 %32, 1
  store i32 %33, i32* %__y, align 4
  br label %9

  ; <label>:34                                      ; preds = %9
  br label %35

  ; <label>:35                                      ; preds = %34
  %36 = load i32, i32* %__z, align 4
  %37 = add i32 %36, 1
  store i32 %37, i32* %__z, align 4
  br label %4

  ; <label>:38                                      ; preds = %4
  ret void
}

declare void @__dummy_kernel(i64, i64, i64) #1

attributes #0 = { uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-ma  th"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="fal  se" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"PACXX"}
!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.vectorize.enable", i1 true}
)");
}


namespace pacxx
{
  namespace v2
  {
    NativeBackend::NativeBackend() : _composite(std::make_unique<llvm::Module>("pacxx-link", llvm::getGlobalContext())),
                                     _linker(_composite.get()),
                                     _pmInitialized(false){ }

    NativeBackend::~NativeBackend() {}

    llvm::Module* NativeBackend::compile(llvm::Module &M) {

        std::string error;
        std::error_code EC;

        linkInModule(M);
        llvm::Module *TheModule = _composite.get();

        EngineBuilder builder{std::move(_composite)};

        builder.setErrorStr(&error);

        builder.setEngineKind(EngineKind::JIT);

        builder.setMCJITMemoryManager(
                std::unique_ptr<RTDyldMemoryManager>(
                        static_cast<RTDyldMemoryManager*>(new SectionMemoryManager())));

      _JITEngine = builder.create();
      if (!_JITEngine) {
        throw new common::generic_exception(error);
      }

      TheModule->setDataLayout(_JITEngine->getDataLayout());

      // TODO remove
      llvm::raw_fd_ostream OS("moduleBeforePass", EC, llvm::sys::fs::F_None);
      TheModule->print(OS, nullptr);

      applyPasses(*TheModule);

      __verbose("applied pass");

      //TODO remove
      llvm::raw_fd_ostream OS1("moduleAfterPass", EC, llvm::sys::fs::F_None);
      TheModule->print(OS1, nullptr);

      _JITEngine->finalizeObject();

      return TheModule;
    }

    void* NativeBackend::getKernelFptr(llvm::Module *module, const std::string name) {
        llvm::Function *kernel = module->getFunction("__wrapped__"+name);
        //get the kernel wrapper function from the module
        return _JITEngine->getPointerToFunction(kernel);
    }

    void NativeBackend::linkInModule(llvm::Module& M) {
        std::unique_ptr<llvm::Module> functionModule = NativeBackend::createModule(_composite->getContext(), native_loop_ir);
        _linker.linkInModule(functionModule.get(), llvm::Linker::Flags::None, nullptr);
        _linker.linkInModule(&M, llvm::Linker::Flags::None, nullptr);
        _composite->setTargetTriple(sys::getProcessTriple());
    }

    std::unique_ptr<llvm::Module> NativeBackend::createModule(llvm::LLVMContext& Context, const std::string IR) {
        llvm::SMDiagnostic Err;
        llvm::MemoryBufferRef buffer(IR, "loop-buffer");
        std::unique_ptr<llvm::Module> Result = llvm::parseIR(buffer, Err, Context);
        if (!Result)
            Err.print("createModule", llvm::errs());
        Result->materializeMetadata();
        return Result;
    }

    void NativeBackend::applyPasses(llvm::Module& M) {

        string Error;

        if(!_target)
           _target = TargetRegistry::lookupTarget(M.getTargetTriple(), Error);
        if(!_target)
            throw common::generic_exception(Error);

        if(!_pmInitialized) {
            _PM.add(createPACXXNativeLinker());
            _PM.add(createLoopSimplifyPass());
            _PM.add(createLoopInstSimplifyPass());
            _PM.add(createLoopStrengthReducePass());
            _PM.add(createDeadCodeEliminationPass());
            _PM.add(createDeadInstEliminationPass());
            _PM.add(createDeadStoreEliminationPass());
            _pmInitialized = true;
        }
        _PM.run(M);
    }

    llvm::legacy::PassManager& NativeBackend::getPassManager() { return _PM; }

  }
}
