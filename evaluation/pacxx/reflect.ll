; ModuleID = 'kernel.ll'
source_filename = "llvm-link"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: alwaysinline
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0

; Function Attrs: alwaysinline
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0

; Function Attrs: alwaysinline
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: alwaysinline
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #0

; Function Attrs: alwaysinline
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #0

; Function Attrs: alwaysinline
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y() #0

attributes #0 = { alwaysinline }

!llvm.ident = !{!0, !0}
!nvvm.annotations = !{!1}

!0 = !{!"clang version 5.0.0 (https://lklein14@bitbucket.org/mhaidl/clang_v2.git 437c6114787c7112dd7976fef5df73c494d66578) (https://lklein14@bitbucket.org/mhaidl/llvm_v2.git faaf5abbec4de280db0baf4c3d4226183d57f591)"}
!1 = !{void (i8, i32*, float, float, float, float)* undef, !"kernel", i32 1}
