; ModuleID = './tmp-170202-1253-ZN7zhg/kernel.ll'
source_filename = "llvm-link"
target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-unknown-unknown"

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #0

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #0

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y() #0

; Function Attrs: noinline nounwind uwtable
define void @"_ZN5pacxx2v213genericKernelILm0EZ4mainE3$_0JPiffffEEEvT0_DpT1_"(i8 %callable.coerce, i32* nocapture %args, float %args1, float %args3, float %args5, float %args7) #1 !kernel_arg_addr_space !2 !kernel_arg_access_qual !3 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2, !range !6
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #2, !range !7
  %mul.i = mul nuw nsw i32 %1, %0
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2, !range !8
  %add.i = add nuw nsw i32 %mul.i, %2
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #2, !range !6
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #2, !range !7
  %mul6.i = mul nuw nsw i32 %4, %3
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #2, !range !8
  %add8.i = add nuw nsw i32 %mul6.i, %5
  %6 = or i32 %add.i, %add8.i
  %7 = and i32 %6, 268434432
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %if.end.i, label %"_ZZ4mainENK3$_0clEPiffff.exit"

if.end.i:                                         ; preds = %entry
  %conv.i = uitofp i32 %add.i to float
  %div.i = fmul float %conv.i, 9.765625e-04
  %sub.i = fsub float %args3, %args1
  %mul10.i = fmul float %sub.i, %div.i
  %add11.i = fadd float %mul10.i, %args1
  %conv12.i = uitofp i32 %add8.i to float
  %div13.i = fmul float %conv12.i, 9.765625e-04
  %sub14.i = fsub float %args7, %args5
  %mul15.i = fmul float %sub14.i, %div13.i
  %add16.i = fadd float %mul15.i, %args5
  br label %while.body.i

while.body.i:                                     ; preds = %while.body.i, %if.end.i
  %dec5.i = phi i32 [ 999, %if.end.i ], [ %dec.i, %while.body.i ]
  %zi2.04.i = phi float [ 0.000000e+00, %if.end.i ], [ %mul25.i, %while.body.i ]
  %zr2.03.i = phi float [ 0.000000e+00, %if.end.i ], [ %mul24.i, %while.body.i ]
  %zr.02.i = phi float [ 0.000000e+00, %if.end.i ], [ %add23.i, %while.body.i ]
  %zi.01.i = phi float [ 0.000000e+00, %if.end.i ], [ %add21.i, %while.body.i ]
  %mul19.i = fmul float %zi.01.i, %zr.02.i
  %add20.i = fadd float %mul19.i, %mul19.i
  %add21.i = fadd float %add16.i, %add20.i
  %sub22.i = fsub float %zr2.03.i, %zi2.04.i
  %add23.i = fadd float %add11.i, %sub22.i
  %mul24.i = fmul float %add23.i, %add23.i
  %mul25.i = fmul float %add21.i, %add21.i
  %dec.i = add i32 %dec5.i, -1
  %tobool.i = icmp ne i32 %dec.i, 0
  %add17.i = fadd float %mul24.i, %mul25.i
  %cmp18.i = fcmp olt float %add17.i, 4.000000e+00
  %9 = and i1 %tobool.i, %cmp18.i
  br i1 %9, label %while.body.i, label %while.end.i

while.end.i:                                      ; preds = %while.body.i
  %sub28.i = sub i32 1001, %dec5.i
  %10 = uitofp i32 %sub28.i to float
  %.op = fmul float %10, 5.000000e+00
  %11 = fptosi float %.op to i32
  %conv32.i = select i1 %tobool.i, i32 %11, i32 0
  %mul33.i = shl i32 %add8.i, 10
  %add34.i = add i32 %mul33.i, %add.i
  %idxprom.i = zext i32 %add34.i to i64
  %arrayidx.i = getelementptr inbounds i32, i32* %args, i64 %idxprom.i
  store i32 %conv32.i, i32* %arrayidx.i, align 4
  br label %"_ZZ4mainENK3$_0clEPiffff.exit"

"_ZZ4mainENK3$_0clEPiffff.exit":                  ; preds = %entry, %while.end.i
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!llvm.ident = !{!0, !0}
!nvvm.annotations = !{!1}

!0 = !{!"clang version 5.0.0 (https://lklein14@bitbucket.org/mhaidl/clang_v2.git 437c6114787c7112dd7976fef5df73c494d66578) (https://lklein14@bitbucket.org/mhaidl/llvm_v2.git faaf5abbec4de280db0baf4c3d4226183d57f591)"}
!1 = !{void (i8, i32*, float, float, float, float)* @"_ZN5pacxx2v213genericKernelILm0EZ4mainE3$_0JPiffffEEEvT0_DpT1_", !"kernel", i32 1}
!2 = !{i32 0, i32 0, i32 0, i32 0, i32 0, i32 0}
!3 = !{!"none", !"none", !"none", !"none", !"none", !"none"}
!4 = !{!"(lambda at mandel.cpp:64:15)", !"int*", !"float", !"float", !"float", !"float"}
!5 = !{!"", !"", !"", !"", !"", !""}
!6 = !{i32 0, i32 65535}
!7 = !{i32 1, i32 1025}
!8 = !{i32 0, i32 1024}
