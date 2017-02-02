; ModuleID = 'pacxx-link'
source_filename = "pacxx-link"
target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: uwtable
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

; <label>:4:                                      ; preds = %27, %0
  %5 = load i32, i32* %__z, align 4
  %6 = load i32, i32* %3, align 4
  %7 = icmp ult i32 %5, %6
  br i1 %7, label %8, label %30

; <label>:8:                                      ; preds = %4
  store i32 0, i32* %__y, align 4
  br label %9

; <label>:9:                                      ; preds = %23, %8
  %10 = load i32, i32* %__y, align 4
  %11 = load i32, i32* %2, align 4
  %12 = icmp ult i32 %10, %11
  br i1 %12, label %13, label %26

; <label>:13:                                     ; preds = %9
  store i32 0, i32* %__x, align 4
  br label %14

; <label>:14:                                     ; preds = %19, %13
  %15 = load i32, i32* %__x, align 4
  %16 = load i32, i32* %1, align 4
  %17 = icmp ult i32 %15, %16
  br i1 %17, label %18, label %22

; <label>:18:                                     ; preds = %14
  call void @__dummy_kernel()
  br label %19

; <label>:19:                                     ; preds = %18
  %20 = load i32, i32* %__x, align 4
  %21 = add i32 %20, 1
  store i32 %21, i32* %__x, align 4
  br label %14, !llvm.loop !3

; <label>:22:                                     ; preds = %14
  br label %23

; <label>:23:                                     ; preds = %22
  %24 = load i32, i32* %__y, align 4
  %25 = add i32 %24, 1
  store i32 %25, i32* %__y, align 4
  br label %9

; <label>:26:                                     ; preds = %9
  br label %27

; <label>:27:                                     ; preds = %26
  %28 = load i32, i32* %__z, align 4
  %29 = add i32 %28, 1
  store i32 %29, i32* %__z, align 4
  br label %4

; <label>:30:                                     ; preds = %4
  ret void
}

declare void @__dummy_kernel() #1

; Function Attrs: noinline nounwind uwtable
define void @"_ZN5pacxx2v213genericKernelILm0EZ4mainE3$_0JPiffffEEEvT0_DpT1_"(i8 %callable.coerce, i32* nocapture %args, float %args1, float %args3, float %args5, float %args7) #2 !kernel_arg_addr_space !5 !kernel_arg_access_qual !6 !kernel_arg_type !7 !kernel_arg_base_type !7 !kernel_arg_type_qual !8 {
entry:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4, !range !9
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4, !range !10
  %mul.i = mul nuw nsw i32 %1, %0
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !range !11
  %add.i = add nuw nsw i32 %mul.i, %2
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #4, !range !9
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #4, !range !10
  %mul6.i = mul nuw nsw i32 %4, %3
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #4, !range !11
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

"_ZZ4mainENK3$_0clEPiffff.exit":                  ; preds = %while.end.i, %entry
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #3

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #3

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #3

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y() #3

attributes #0 = { uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-ma  th"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="fal  se" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2, +avx" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind readnone }
attributes #4 = { nounwind }

!llvm.ident = !{!0, !1, !1}
!nvvm.annotations = !{!2}

!0 = !{!"PACXX"}
!1 = !{!"clang version 5.0.0 (https://lklein14@bitbucket.org/mhaidl/clang_v2.git 437c6114787c7112dd7976fef5df73c494d66578) (https://lklein14@bitbucket.org/mhaidl/llvm_v2.git faaf5abbec4de280db0baf4c3d4226183d57f591)"}
!2 = !{void (i8, i32*, float, float, float, float)* @"_ZN5pacxx2v213genericKernelILm0EZ4mainE3$_0JPiffffEEEvT0_DpT1_", !"kernel", i32 1}
!3 = distinct !{!3, !4}
!4 = !{!"llvm.loop.vectorize.enable", i1 false}
!5 = !{i32 0, i32 0, i32 0, i32 0, i32 0, i32 0}
!6 = !{!"none", !"none", !"none", !"none", !"none", !"none"}
!7 = !{!"(lambda at mandel.cpp:64:15)", !"int*", !"float", !"float", !"float", !"float"}
!8 = !{!"", !"", !"", !"", !"", !""}
!9 = !{i32 0, i32 65535}
!10 = !{i32 1, i32 1025}
!11 = !{i32 0, i32 1024}
