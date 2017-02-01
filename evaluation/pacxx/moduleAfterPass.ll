; ModuleID = 'pacxx-link'
source_filename = "pacxx-link"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: norecurse nounwind
define void @"__wrapped___ZN5pacxx2v213genericKernelILm0EZ4mainE3$_0JPiffffEEEvT0_DpT1_"(i32 %bidx, i32 %bidy, i32 %bidz, i32 %maxblockx, i32 %maxblocky, i32 %maxblockz, i32 %maxidx, i32 %maxidy, i32 %maxidz, i32 %sm_size, i8* nocapture readonly %args) local_unnamed_addr #0 {
constructArgs:
  %0 = getelementptr inbounds i8, i8* %args, i64 8
  %1 = bitcast i8* %0 to i32**
  %args3 = load i32*, i32** %1, align 8
  %2 = getelementptr inbounds i8, i8* %args, i64 16
  %3 = bitcast i8* %2 to float*
  %args14 = load float, float* %3, align 8
  %4 = getelementptr inbounds i8, i8* %args, i64 24
  %5 = bitcast i8* %4 to float*
  %args5 = load float, float* %5, align 8
  %6 = icmp eq i32 %maxidz, 0
  br i1 %6, label %"__vectorized__foo___ZN5pacxx2v213genericKernelILm0EZ4mainE3$_0JPiffffEEEvT0_DpT1_.exit", label %.preheader8.lr.ph

.preheader8.lr.ph:                                ; preds = %constructArgs
  %7 = getelementptr inbounds i8, i8* %args, i64 28
  %8 = bitcast i8* %7 to float*
  %args7 = load float, float* %8, align 8
  %9 = getelementptr inbounds i8, i8* %args, i64 20
  %10 = bitcast i8* %9 to float*
  %args35 = load float, float* %10, align 8
  %11 = icmp eq i32 %maxidy, 0
  %cmp.i9 = icmp sgt i32 %maxidy, 4
  %mul6.i.i.i = mul nuw nsw i32 %maxidy, %bidy
  %mul.i.i.i = mul nuw nsw i32 %maxidx, %bidx
  %sub.i.i.i = fsub float %args35, %args14
  %sub14.i.i.i = fsub float %args7, %args5
  %12 = insertelement <4 x float> undef, float %sub.i.i.i, i32 0
  %13 = shufflevector <4 x float> %12, <4 x float> undef, <4 x i32> zeroinitializer
  %14 = insertelement <4 x float> undef, float %args14, i32 0
  %15 = shufflevector <4 x float> %14, <4 x float> undef, <4 x i32> zeroinitializer
  %16 = zext i32 %mul.i.i.i to i64
  %wide.trip.count = zext i32 %maxidx to i64
  br label %.preheader8

.preheader8:                                      ; preds = %._crit_edge14, %.preheader8.lr.ph
  %__z.i.015 = phi i32 [ 0, %.preheader8.lr.ph ], [ %61, %._crit_edge14 ]
  br i1 %11, label %._crit_edge14, label %loop-header.i.preheader.preheader

loop-header.i.preheader.preheader:                ; preds = %.preheader8
  br label %loop-header.i.preheader

loop-header.i.preheader:                          ; preds = %loop-header.i.preheader.preheader, %._crit_edge
  %__y.i.013 = phi i32 [ %60, %._crit_edge ], [ 0, %loop-header.i.preheader.preheader ]
  br i1 %cmp.i9, label %loop-body.i.lr.ph, label %.preheader

loop-body.i.lr.ph:                                ; preds = %loop-header.i.preheader
  %add8.i..i.i = add nuw nsw i32 %__y.i.013, %mul6.i.i.i
  %17 = insertelement <4 x i32> undef, i32 %add8.i..i.i, i32 0
  %18 = shufflevector <4 x i32> %17, <4 x i32> undef, <4 x i32> zeroinitializer
  %conv12.i..i.i = uitofp i32 %add8.i..i.i to float
  %div13.i..i.i = fmul float %conv12.i..i.i, 9.765625e-04
  %mul15.i..i.i = fmul float %sub14.i.i.i, %div13.i..i.i
  %add16.i..i.i = fadd float %args5, %mul15.i..i.i
  %19 = insertelement <4 x float> undef, float %add16.i..i.i, i32 0
  %20 = shufflevector <4 x float> %19, <4 x float> undef, <4 x i32> zeroinitializer
  %mul33.i..i.i = shl i32 %add8.i..i.i, 10
  br label %loop-body.i

.preheader.loopexit:                              ; preds = %loop-header.i.backedge
  br label %.preheader

.preheader:                                       ; preds = %.preheader.loopexit, %loop-header.i.preheader
  %__x.i.0.lcssa = phi i32 [ 0, %loop-header.i.preheader ], [ %"increment loop var.i11", %.preheader.loopexit ]
  %21 = icmp ult i32 %__x.i.0.lcssa, %maxidx
  br i1 %21, label %.lr.ph, label %._crit_edge

.lr.ph:                                           ; preds = %.preheader
  %add8.i.i.i = add nuw nsw i32 %__y.i.013, %mul6.i.i.i
  %conv12.i.i.i = uitofp i32 %add8.i.i.i to float
  %div13.i.i.i = fmul float %conv12.i.i.i, 9.765625e-04
  %mul15.i.i.i = fmul float %sub14.i.i.i, %div13.i.i.i
  %add16.i.i.i = fadd float %args5, %mul15.i.i.i
  %mul33.i.i.i = shl i32 %add8.i.i.i, 10
  %22 = zext i32 %__x.i.0.lcssa to i64
  br label %51

loop-body.i:                                      ; preds = %loop-body.i.lr.ph, %loop-header.i.backedge
  %"increment loop var.i11" = phi i32 [ 4, %loop-body.i.lr.ph ], [ %"increment loop var.i", %loop-header.i.backedge ]
  %__x.i.010 = phi i32 [ 0, %loop-body.i.lr.ph ], [ %"increment loop var.i11", %loop-header.i.backedge ]
  %add.i..i.i = add nuw nsw i32 %__x.i.010, %mul.i.i.i
  %23 = insertelement <4 x i32> undef, i32 %add.i..i.i, i32 0
  %24 = shufflevector <4 x i32> %23, <4 x i32> undef, <4 x i32> zeroinitializer
  %25 = add <4 x i32> %24, <i32 0, i32 1, i32 2, i32 3>
  %26 = or <4 x i32> %25, %18
  %27 = and <4 x i32> %26, <i32 268434432, i32 268434432, i32 268434432, i32 268434432>
  %28 = icmp eq <4 x i32> %27, zeroinitializer
  %conv.i..i.i = uitofp i32 %add.i..i.i to float
  %29 = insertelement <4 x float> undef, float %conv.i..i.i, i32 0
  %30 = shufflevector <4 x float> %29, <4 x float> undef, <4 x i32> zeroinitializer
  %31 = fadd <4 x float> %30, <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>
  %div.i..i.i = fmul <4 x float> %31, <float 9.765625e-04, float 9.765625e-04, float 9.765625e-04, float 9.765625e-04>
  %mul10.i..i.i = fmul <4 x float> %13, %div.i..i.i
  %add11.i..i.i = fadd <4 x float> %15, %mul10.i..i.i
  br label %while.body.i..i.i

while.body.i..i.i:                                ; preds = %while.body.i..i.i, %loop-body.i
  %dec5.i..i.i = phi i32 [ 999, %loop-body.i ], [ %dec.i..i.i, %while.body.i..i.i ]
  %zi2.04.i..i.i = phi <4 x float> [ zeroinitializer, %loop-body.i ], [ %mul25.i..i.i, %while.body.i..i.i ]
  %zr2.03.i..i.i = phi <4 x float> [ zeroinitializer, %loop-body.i ], [ %mul24.i..i.i, %while.body.i..i.i ]
  %zr.02.i..i.i = phi <4 x float> [ zeroinitializer, %loop-body.i ], [ %add23.i..i.i, %while.body.i..i.i ]
  %zi.01.i..i.i = phi <4 x float> [ zeroinitializer, %loop-body.i ], [ %add21.i..i.i, %while.body.i..i.i ]
  %loopMaskPhi..i.i = phi <4 x i1> [ %28, %loop-body.i ], [ %34, %while.body.i..i.i ]
  %loopExitMaskPhi..i.i = phi <4 x i1> [ zeroinitializer, %loop-body.i ], [ %loopMaskUpdate..i.i, %while.body.i..i.i ]
  %mul19.i..i.i = fmul <4 x float> %zr.02.i..i.i, %zi.01.i..i.i
  %add20.i..i.i = fadd <4 x float> %mul19.i..i.i, %mul19.i..i.i
  %add21.i..i.i = fadd <4 x float> %20, %add20.i..i.i
  %sub22.i..i.i = fsub <4 x float> %zr2.03.i..i.i, %zi2.04.i..i.i
  %add23.i..i.i = fadd <4 x float> %add11.i..i.i, %sub22.i..i.i
  %mul24.i..i.i = fmul <4 x float> %add23.i..i.i, %add23.i..i.i
  %mul25.i..i.i = fmul <4 x float> %add21.i..i.i, %add21.i..i.i
  %dec.i..i.i = add i32 %dec5.i..i.i, -1
  %tobool.i..i.i = icmp ne i32 %dec.i..i.i, 0
  %add17.i..i.i = fadd <4 x float> %mul24.i..i.i, %mul25.i..i.i
  %cmp18.i..i.i = fcmp olt <4 x float> %add17.i..i.i, <float 4.000000e+00, float 4.000000e+00, float 4.000000e+00, float 4.000000e+00>
  %32 = select i1 %tobool.i..i.i, <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x i1> zeroinitializer
  %33 = and <4 x i1> %cmp18.i..i.i, %32
  %34 = and <4 x i1> %33, %loopMaskPhi..i.i
  %35 = xor <4 x i1> %33, <i1 true, i1 true, i1 true, i1 true>
  %combinedLoopExitMask..i.i = and <4 x i1> %loopMaskPhi..i.i, %35
  %loopMaskUpdate..i.i = or <4 x i1> %combinedLoopExitMask..i.i, %loopExitMaskPhi..i.i
  %36 = sext <4 x i1> %34 to <4 x i32>
  %37 = bitcast <4 x i32> %36 to i128
  %38 = icmp eq i128 %37, 0
  br i1 %38, label %while.end.i..i.i, label %while.body.i..i.i

while.end.i..i.i:                                 ; preds = %while.body.i..i.i
  %39 = insertelement <4 x i32> undef, i32 %dec5.i..i.i, i32 0
  %40 = shufflevector <4 x i32> %39, <4 x i32> undef, <4 x i32> zeroinitializer
  %sub28.i..i.i = sub <4 x i32> <i32 1001, i32 1001, i32 1001, i32 1001>, %40
  %41 = uitofp <4 x i32> %sub28.i..i.i to <4 x float>
  %.op..i.i = fmul <4 x float> %41, <float 5.000000e+00, float 5.000000e+00, float 5.000000e+00, float 5.000000e+00>
  %42 = fptosi <4 x float> %.op..i.i to <4 x i32>
  %conv32.i..i.i = select i1 %tobool.i..i.i, <4 x i32> %42, <4 x i32> zeroinitializer
  %add34.i..i.i = add i32 %add.i..i.i, %mul33.i..i.i
  %idxprom.i..i.i = zext i32 %add34.i..i.i to i64
  %43 = getelementptr i32, i32* %args3, i64 %idxprom.i..i.i
  %unpackW210.i.i = extractelement <4 x i32> %conv32.i..i.i, i32 1
  %unpackW311.i.i = extractelement <4 x i32> %conv32.i..i.i, i32 2
  %unpackW412.i.i = extractelement <4 x i32> %conv32.i..i.i, i32 3
  %44 = getelementptr i32, i32* %43, i64 1
  %45 = getelementptr i32, i32* %43, i64 2
  %46 = getelementptr i32, i32* %43, i64 3
  %47 = extractelement <4 x i1> %loopMaskUpdate..i.i, i32 0
  br i1 %47, label %casc.exec0.i.i, label %casc.if1.i.i

casc.exec0.i.i:                                   ; preds = %while.end.i..i.i
  %unpackW9.i.i = extractelement <4 x i32> %conv32.i..i.i, i32 0
  store i32 %unpackW9.i.i, i32* %43, align 4
  br label %casc.if1.i.i

casc.if1.i.i:                                     ; preds = %casc.exec0.i.i, %while.end.i..i.i
  %48 = extractelement <4 x i1> %loopMaskUpdate..i.i, i32 1
  br i1 %48, label %casc.exec1.i.i, label %casc.if2.i.i

casc.exec1.i.i:                                   ; preds = %casc.if1.i.i
  store i32 %unpackW210.i.i, i32* %44, align 4
  br label %casc.if2.i.i

casc.if2.i.i:                                     ; preds = %casc.exec1.i.i, %casc.if1.i.i
  %49 = extractelement <4 x i1> %loopMaskUpdate..i.i, i32 2
  br i1 %49, label %casc.exec2.i.i, label %casc.if3.i.i

casc.exec2.i.i:                                   ; preds = %casc.if2.i.i
  store i32 %unpackW311.i.i, i32* %45, align 4
  br label %casc.if3.i.i

casc.if3.i.i:                                     ; preds = %casc.exec2.i.i, %casc.if2.i.i
  %50 = extractelement <4 x i1> %loopMaskUpdate..i.i, i32 3
  br i1 %50, label %casc.exec3.i.i, label %loop-header.i.backedge

casc.exec3.i.i:                                   ; preds = %casc.if3.i.i
  store i32 %unpackW412.i.i, i32* %46, align 4
  br label %loop-header.i.backedge

loop-header.i.backedge:                           ; preds = %casc.exec3.i.i, %casc.if3.i.i
  %"increment loop var.i" = add i32 %"increment loop var.i11", 4
  %cmp.i = icmp slt i32 %"increment loop var.i", %maxidy
  br i1 %cmp.i, label %loop-body.i, label %.preheader.loopexit

; <label>:51:                                     ; preds = %"_ZN5pacxx2v213genericKernelILm0EZ4mainE3$_0JPiffffEEEvT0_DpT1_.exit.i", %.lr.ph
  %indvars.iv = phi i64 [ %22, %.lr.ph ], [ %indvars.iv.next, %"_ZN5pacxx2v213genericKernelILm0EZ4mainE3$_0JPiffffEEEvT0_DpT1_.exit.i" ]
  %52 = add nuw nsw i64 %indvars.iv, %16
  %53 = trunc i64 %52 to i32
  %54 = or i32 %53, %add8.i.i.i
  %55 = and i32 %54, 268434432
  %56 = icmp eq i32 %55, 0
  br i1 %56, label %if.end.i.i.i, label %"_ZN5pacxx2v213genericKernelILm0EZ4mainE3$_0JPiffffEEEvT0_DpT1_.exit.i"

if.end.i.i.i:                                     ; preds = %51
  %conv.i.i.i = uitofp i32 %53 to float
  %div.i.i.i = fmul float %conv.i.i.i, 9.765625e-04
  %mul10.i.i.i = fmul float %sub.i.i.i, %div.i.i.i
  %add11.i.i.i = fadd float %args14, %mul10.i.i.i
  br label %while.body.i.i.i

while.body.i.i.i:                                 ; preds = %while.body.i.i.i, %if.end.i.i.i
  %dec5.i.i.i = phi i32 [ 999, %if.end.i.i.i ], [ %dec.i.i.i, %while.body.i.i.i ]
  %zi2.04.i.i.i = phi float [ 0.000000e+00, %if.end.i.i.i ], [ %mul25.i.i.i, %while.body.i.i.i ]
  %zr2.03.i.i.i = phi float [ 0.000000e+00, %if.end.i.i.i ], [ %mul24.i.i.i, %while.body.i.i.i ]
  %zr.02.i.i.i = phi float [ 0.000000e+00, %if.end.i.i.i ], [ %add23.i.i.i, %while.body.i.i.i ]
  %zi.01.i.i.i = phi float [ 0.000000e+00, %if.end.i.i.i ], [ %add21.i.i.i, %while.body.i.i.i ]
  %mul19.i.i.i = fmul float %zr.02.i.i.i, %zi.01.i.i.i
  %add20.i.i.i = fadd float %mul19.i.i.i, %mul19.i.i.i
  %add21.i.i.i = fadd float %add16.i.i.i, %add20.i.i.i
  %sub22.i.i.i = fsub float %zr2.03.i.i.i, %zi2.04.i.i.i
  %add23.i.i.i = fadd float %add11.i.i.i, %sub22.i.i.i
  %mul24.i.i.i = fmul float %add23.i.i.i, %add23.i.i.i
  %mul25.i.i.i = fmul float %add21.i.i.i, %add21.i.i.i
  %dec.i.i.i = add nsw i32 %dec5.i.i.i, -1
  %tobool.i.i.i = icmp ne i32 %dec.i.i.i, 0
  %add17.i.i.i = fadd float %mul24.i.i.i, %mul25.i.i.i
  %cmp18.i.i.i = fcmp olt float %add17.i.i.i, 4.000000e+00
  %57 = and i1 %tobool.i.i.i, %cmp18.i.i.i
  br i1 %57, label %while.body.i.i.i, label %while.end.i.i.i

while.end.i.i.i:                                  ; preds = %while.body.i.i.i
  %sub28.i.i.i = sub i32 1001, %dec5.i.i.i
  %58 = uitofp i32 %sub28.i.i.i to float
  %.op.i.i = fmul float %58, 5.000000e+00
  %59 = fptosi float %.op.i.i to i32
  %conv32.i.i.i = select i1 %tobool.i.i.i, i32 %59, i32 0
  %add34.i.i.i = add i32 %53, %mul33.i.i.i
  %idxprom.i.i.i = zext i32 %add34.i.i.i to i64
  %arrayidx.i.i.i = getelementptr inbounds i32, i32* %args3, i64 %idxprom.i.i.i
  store i32 %conv32.i.i.i, i32* %arrayidx.i.i.i, align 4
  br label %"_ZN5pacxx2v213genericKernelILm0EZ4mainE3$_0JPiffffEEEvT0_DpT1_.exit.i"

"_ZN5pacxx2v213genericKernelILm0EZ4mainE3$_0JPiffffEEEvT0_DpT1_.exit.i": ; preds = %while.end.i.i.i, %51
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond, label %._crit_edge.loopexit, label %51, !llvm.loop !3

._crit_edge.loopexit:                             ; preds = %"_ZN5pacxx2v213genericKernelILm0EZ4mainE3$_0JPiffffEEEvT0_DpT1_.exit.i"
  br label %._crit_edge

._crit_edge:                                      ; preds = %._crit_edge.loopexit, %.preheader
  %60 = add nuw i32 %__y.i.013, 1
  %exitcond17 = icmp eq i32 %60, %maxidy
  br i1 %exitcond17, label %._crit_edge14.loopexit, label %loop-header.i.preheader

._crit_edge14.loopexit:                           ; preds = %._crit_edge
  br label %._crit_edge14

._crit_edge14:                                    ; preds = %._crit_edge14.loopexit, %.preheader8
  %61 = add nuw i32 %__z.i.015, 1
  %exitcond18 = icmp eq i32 %61, %maxidz
  br i1 %exitcond18, label %"__vectorized__foo___ZN5pacxx2v213genericKernelILm0EZ4mainE3$_0JPiffffEEEvT0_DpT1_.exit.loopexit", label %.preheader8

"__vectorized__foo___ZN5pacxx2v213genericKernelILm0EZ4mainE3$_0JPiffffEEEvT0_DpT1_.exit.loopexit": ; preds = %._crit_edge14
  br label %"__vectorized__foo___ZN5pacxx2v213genericKernelILm0EZ4mainE3$_0JPiffffEEEvT0_DpT1_.exit"

"__vectorized__foo___ZN5pacxx2v213genericKernelILm0EZ4mainE3$_0JPiffffEEEvT0_DpT1_.exit": ; preds = %"__vectorized__foo___ZN5pacxx2v213genericKernelILm0EZ4mainE3$_0JPiffffEEEvT0_DpT1_.exit.loopexit", %constructArgs
  ret void
}

attributes #0 = { norecurse nounwind }

!llvm.ident = !{!0, !1, !1}
!nvvm.annotations = !{!2}

!0 = !{!"PACXX"}
!1 = !{!"clang version 5.0.0 (https://lklein14@bitbucket.org/mhaidl/clang_v2.git 2904b703a745e9588c3f2b172c9e3178db89e088) (https://lklein14@bitbucket.org/mhaidl/llvm_v2.git abe9a99c1000ee259fd76b9e00f4fd47498e0fea)"}
!2 = distinct !{null, !"kernel", i32 1}
!3 = distinct !{!3, !4}
!4 = !{!"llvm.loop.vectorize.enable", i1 false}
