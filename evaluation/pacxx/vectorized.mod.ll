; ModuleID = 'pacxx-link'
source_filename = "pacxx-link"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
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

; <label>:4:                                      ; preds = %24, %0
  %5 = load i32, i32* %__z, align 4
  %6 = load i32, i32* %3, align 4
  %7 = icmp ult i32 %5, %6
  br i1 %7, label %8, label %27

; <label>:8:                                      ; preds = %4
  store i32 0, i32* %__y, align 4
  br label %9

; <label>:9:                                      ; preds = %21, %8
  %10 = load i32, i32* %__y, align 4
  %11 = load i32, i32* %2, align 4
  %12 = icmp ult i32 %10, %11
  br i1 %12, label %13, label %24

; <label>:13:                                     ; preds = %9
  store i32 0, i32* %__x, align 4
  br label %14

; <label>:14:                                     ; preds = %18, %13
  %15 = load i32, i32* %__x, align 4
  %16 = load i32, i32* %1, align 4
  %17 = icmp ult i32 %15, %16
  br i1 %17, label %18, label %21

; <label>:18:                                     ; preds = %14
  call void @__dummy_kernel()
  %19 = load i32, i32* %__x, align 4
  %20 = add i32 %19, 1
  store i32 %20, i32* %__x, align 4
  br label %14, !llvm.loop !3

; <label>:21:                                     ; preds = %14
  %22 = load i32, i32* %__y, align 4
  %23 = add i32 %22, 1
  store i32 %23, i32* %__y, align 4
  br label %9

; <label>:24:                                     ; preds = %9
  %25 = load i32, i32* %__z, align 4
  %26 = add i32 %25, 1
  store i32 %26, i32* %__z, align 4
  br label %4

; <label>:27:                                     ; preds = %4
  ret void
}

declare void @__dummy_kernel() #1

; Function Attrs: noinline nounwind uwtable
define void @"_ZN5pacxx2v213genericKernelILm0EZ4mainE3$_0JPiffffEEEvT0_DpT1_"(i8 %callable.coerce, i32* nocapture %args, float %args1, float %args3, float %args5, float %args7) #2 !kernel_arg_addr_space !5 !kernel_arg_access_qual !6 !kernel_arg_type !7 !kernel_arg_base_type !7 !kernel_arg_type_qual !8 {
IdCalc:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #4, !range !9
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #4, !range !10
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #4, !range !11
  %mul6.i = mul nuw nsw i32 %2, %1
  %add8.i = add nuw nsw i32 %mul6.i, %0
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !range !9
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4, !range !10
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4, !range !11
  %mul.i = mul nuw nsw i32 %5, %4
  %add.i = add nuw nsw i32 %mul.i, %3
  %6 = or i32 %add.i, %add8.i
  %7 = and i32 %6, 268434432
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %if.end.i, label %"_ZZ4mainENK3$_0clEPiffff.exit"

if.end.i:                                         ; preds = %IdCalc
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
  %dec5.i.lcssa = phi i32 [ %dec5.i, %while.body.i ]
  %tobool.i.lcssa = phi i1 [ %tobool.i, %while.body.i ]
  %sub28.i = sub i32 1001, %dec5.i.lcssa
  %10 = uitofp i32 %sub28.i to float
  %.op = fmul float %10, 5.000000e+00
  %11 = fptosi float %.op to i32
  %conv32.i = select i1 %tobool.i.lcssa, i32 %11, i32 0
  %mul33.i = shl i32 %add8.i, 10
  %add34.i = add i32 %mul33.i, %add.i
  %idxprom.i = zext i32 %add34.i to i64
  %arrayidx.i = getelementptr inbounds i32, i32* %args, i64 %idxprom.i
  store i32 %conv32.i, i32* %arrayidx.i, align 4
  br label %"_ZZ4mainENK3$_0clEPiffff.exit"

"_ZZ4mainENK3$_0clEPiffff.exit":                  ; preds = %while.end.i, %IdCalc
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

define void @"__vectorized___ZN5pacxx2v213genericKernelILm0EZ4mainE3$_0JPiffffEEEvT0_DpT1_"(i8 %callable.coerce, i32* %args, float %args1, float %args3, float %args5, float %args7) !kernel_arg_addr_space !5 !kernel_arg_access_qual !6 !kernel_arg_type !7 !kernel_arg_base_type !7 !kernel_arg_type_qual !8 {
IdCalc.:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #4, !range !9, !op_uniform !12, !res_uniform !12, !unaligned !12, !same !12
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #4, !range !10, !op_uniform !12, !res_uniform !12, !unaligned !12, !same !12
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #4, !range !11, !op_uniform !12, !res_uniform !12, !unaligned !12, !same !12
  %mul6.i. = mul nuw nsw i32 %2, %1, !op_uniform !12, !res_uniform !12, !unaligned !12, !same !12
  %add8.i. = add nuw nsw i32 %mul6.i., %0, !op_uniform !12, !res_uniform !12, !unaligned !12, !same !12
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !range !9, !op_varying !12, !res_vector !12, !unaligned !12, !consecutive !12
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4, !range !10, !op_uniform !12, !res_uniform !12, !unaligned !12, !same !12
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4, !range !11, !op_uniform !12, !res_uniform !12, !unaligned !12, !same !12
  %mul.i. = mul nuw nsw i32 %5, %4, !op_uniform !12, !res_uniform !12, !unaligned !12, !same !12
  %add.i. = add nuw nsw i32 %mul.i., %3, !op_varying !12, !res_vector !12, !unaligned !12, !consecutive !12
  %6 = insertelement <4 x i32> undef, i32 %add.i., i32 0, !wfv_pack_unpack !12
  %7 = shufflevector <4 x i32> %6, <4 x i32> undef, <4 x i32> zeroinitializer, !wfv_pack_unpack !12
  %8 = add <4 x i32> %7, <i32 0, i32 1, i32 2, i32 3>, !wfv_pack_unpack !12
  %9 = insertelement <4 x i32> undef, i32 %add8.i., i32 0, !wfv_pack_unpack !12
  %10 = shufflevector <4 x i32> %9, <4 x i32> undef, <4 x i32> zeroinitializer, !wfv_pack_unpack !12
  %11 = or <4 x i32> %8, %10, !op_varying !12, !res_vector !12, !unaligned !12, !random !12
  %12 = and <4 x i32> %11, <i32 268434432, i32 268434432, i32 268434432, i32 268434432>, !op_varying !12, !res_vector !12, !unaligned !12, !random !12
  %13 = icmp eq <4 x i32> %12, zeroinitializer, !op_varying !12, !res_vector !12, !unaligned !12, !random !12, !mask !12
  %14 = xor <4 x i1> %13, <i1 true, i1 true, i1 true, i1 true>, !op_varying !12, !res_vector !12, !mask !12
  br label %"IdCalc._ZZ4mainENK3$_0clEPiffff.exit_crit_edge.", !op_uniform !12

"IdCalc._ZZ4mainENK3$_0clEPiffff.exit_crit_edge.": ; preds = %IdCalc.
  br label %if.end.i., !op_uniform !12

if.end.i.:                                        ; preds = %"IdCalc._ZZ4mainENK3$_0clEPiffff.exit_crit_edge."
  %conv.i. = uitofp i32 %add.i. to float, !op_varying !12, !res_vector !12, !unaligned !12, !consecutive !12
  %15 = insertelement <4 x float> undef, float %conv.i., i32 0, !wfv_pack_unpack !12
  %16 = shufflevector <4 x float> %15, <4 x float> undef, <4 x i32> zeroinitializer, !wfv_pack_unpack !12
  %17 = fadd <4 x float> %16, <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, !wfv_pack_unpack !12
  %div.i. = fmul <4 x float> %17, <float 9.765625e-04, float 9.765625e-04, float 9.765625e-04, float 9.765625e-04>, !op_varying !12, !res_vector !12, !unaligned !12, !random !12
  %sub.i. = fsub float %args3, %args1, !op_uniform !12, !res_uniform !12, !unaligned !12, !same !12
  %18 = insertelement <4 x float> undef, float %sub.i., i32 0, !wfv_pack_unpack !12
  %19 = shufflevector <4 x float> %18, <4 x float> undef, <4 x i32> zeroinitializer, !wfv_pack_unpack !12
  %mul10.i. = fmul <4 x float> %19, %div.i., !op_varying !12, !res_vector !12, !unaligned !12, !random !12
  %20 = insertelement <4 x float> undef, float %args1, i32 0, !wfv_pack_unpack !12
  %21 = shufflevector <4 x float> %20, <4 x float> undef, <4 x i32> zeroinitializer, !wfv_pack_unpack !12
  %add11.i. = fadd <4 x float> %mul10.i., %21, !op_varying !12, !res_vector !12, !unaligned !12, !random !12
  %conv12.i. = uitofp i32 %add8.i. to float, !op_uniform !12, !res_uniform !12, !unaligned !12, !same !12
  %div13.i. = fmul float %conv12.i., 9.765625e-04, !op_uniform !12, !res_uniform !12, !unaligned !12, !same !12
  %sub14.i. = fsub float %args7, %args5, !op_uniform !12, !res_uniform !12, !unaligned !12, !same !12
  %mul15.i. = fmul float %sub14.i., %div13.i., !op_uniform !12, !res_uniform !12, !unaligned !12, !same !12
  %add16.i. = fadd float %mul15.i., %args5, !op_uniform !12, !res_uniform !12, !unaligned !12, !same !12
  br label %while.body.i., !op_uniform !12, !res_uniform !12

while.body.i.:                                    ; preds = %while.body.i.while.body.i_crit_edge., %if.end.i.
  %dec5.i. = phi i32 [ 999, %if.end.i. ], [ %dec.i., %while.body.i.while.body.i_crit_edge. ], !op_uniform !12, !res_uniform !12, !unaligned !12, !same !12
  %zi2.04.i. = phi <4 x float> [ zeroinitializer, %if.end.i. ], [ %mul25.i., %while.body.i.while.body.i_crit_edge. ], !op_uniform !12, !res_vector !12, !unaligned !12, !random !12
  %zr2.03.i. = phi <4 x float> [ zeroinitializer, %if.end.i. ], [ %mul24.i., %while.body.i.while.body.i_crit_edge. ], !op_uniform !12, !res_vector !12, !unaligned !12, !random !12
  %zr.02.i. = phi <4 x float> [ zeroinitializer, %if.end.i. ], [ %add23.i., %while.body.i.while.body.i_crit_edge. ], !op_uniform !12, !res_vector !12, !unaligned !12, !random !12
  %zi.01.i. = phi <4 x float> [ zeroinitializer, %if.end.i. ], [ %add21.i., %while.body.i.while.body.i_crit_edge. ], !op_uniform !12, !res_vector !12, !unaligned !12, !random !12
  %loopMaskPhi. = phi <4 x i1> [ %13, %if.end.i. ], [ %26, %while.body.i.while.body.i_crit_edge. ], !op_uniform !12, !res_vector !12, !mask !12
  %loopExitMaskPhi. = phi <4 x i1> [ zeroinitializer, %if.end.i. ], [ %loopMaskUpdate., %while.body.i.while.body.i_crit_edge. ], !op_uniform !12, !res_vector !12, !mask !12
  %mul19.i. = fmul <4 x float> %zi.01.i., %zr.02.i., !op_varying !12, !res_vector !12, !unaligned !12, !random !12
  %add20.i. = fadd <4 x float> %mul19.i., %mul19.i., !op_varying !12, !res_vector !12, !unaligned !12, !random !12
  %22 = insertelement <4 x float> undef, float %add16.i., i32 0, !wfv_pack_unpack !12
  %23 = shufflevector <4 x float> %22, <4 x float> undef, <4 x i32> zeroinitializer, !wfv_pack_unpack !12
  %add21.i. = fadd <4 x float> %23, %add20.i., !op_varying !12, !res_vector !12, !unaligned !12, !random !12
  %sub22.i. = fsub <4 x float> %zr2.03.i., %zi2.04.i., !op_varying !12, !res_vector !12, !unaligned !12, !random !12
  %add23.i. = fadd <4 x float> %add11.i., %sub22.i., !op_varying !12, !res_vector !12, !unaligned !12, !random !12
  %mul24.i. = fmul <4 x float> %add23.i., %add23.i., !op_varying !12, !res_vector !12, !unaligned !12, !random !12
  %mul25.i. = fmul <4 x float> %add21.i., %add21.i., !op_varying !12, !res_vector !12, !unaligned !12, !random !12
  %dec.i. = add i32 %dec5.i., -1, !op_uniform !12, !res_uniform !12, !unaligned !12, !same !12
  %tobool.i. = icmp ne i32 %dec.i., 0, !op_uniform !12, !res_uniform !12, !unaligned !12, !same !12, !mask !12
  %add17.i. = fadd <4 x float> %mul24.i., %mul25.i., !op_varying !12, !res_vector !12, !unaligned !12, !random !12
  %cmp18.i. = fcmp olt <4 x float> %add17.i., <float 4.000000e+00, float 4.000000e+00, float 4.000000e+00, float 4.000000e+00>, !op_varying !12, !res_vector !12, !unaligned !12, !random !12, !mask !12
  %24 = select i1 %tobool.i., <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x i1> zeroinitializer, !wfv_pack_unpack !12
  %25 = and <4 x i1> %24, %cmp18.i., !op_varying !12, !res_vector !12, !unaligned !12, !random !12, !mask !12
  %26 = and <4 x i1> %loopMaskPhi., %25, !op_varying !12, !res_vector !12, !mask !12
  %27 = xor <4 x i1> %25, <i1 true, i1 true, i1 true, i1 true>, !op_varying !12, !res_vector !12, !mask !12
  %combinedLoopExitMask. = and <4 x i1> %loopMaskPhi., %27, !op_varying !12, !res_vector !12, !mask !12
  %loopMaskUpdate. = or <4 x i1> %loopExitMaskPhi., %combinedLoopExitMask., !op_varying !12, !res_vector !12, !mask !12
  br label %while.body.i.while.body.i_crit_edge., !op_uniform !12

while.body.i.while.body.i_crit_edge.:             ; preds = %while.body.i.
  %28 = sext <4 x i1> %26 to <4 x i32>, !op_varying !12, !res_vector !12, !mask !12
  %29 = bitcast <4 x i32> %28 to i128, !op_uniform !12, !res_uniform !12, !mask !12
  %30 = icmp ne i128 %29, 0, !op_uniform !12, !res_uniform !12
  br i1 %30, label %while.body.i., label %while.end.i., !op_uniform !12, !res_uniform !12

while.end.i.:                                     ; preds = %while.body.i.while.body.i_crit_edge.
  %31 = insertelement <4 x i32> undef, i32 %dec5.i., i32 0, !wfv_pack_unpack !12
  %32 = shufflevector <4 x i32> %31, <4 x i32> undef, <4 x i32> zeroinitializer, !wfv_pack_unpack !12
  %sub28.i. = sub <4 x i32> <i32 1001, i32 1001, i32 1001, i32 1001>, %32, !op_varying !12, !res_vector !12, !unaligned !12, !random !12
  %33 = uitofp <4 x i32> %sub28.i. to <4 x float>, !op_varying !12, !res_vector !12, !unaligned !12, !random !12
  %.op. = fmul <4 x float> %33, <float 5.000000e+00, float 5.000000e+00, float 5.000000e+00, float 5.000000e+00>, !op_varying !12, !res_vector !12, !unaligned !12, !random !12
  %34 = fptosi <4 x float> %.op. to <4 x i32>, !op_varying !12, !res_vector !12, !unaligned !12, !random !12
  %conv32.i. = select i1 %tobool.i., <4 x i32> %34, <4 x i32> zeroinitializer, !op_varying !12, !res_vector !12, !unaligned !12, !random !12
  %mul33.i. = shl i32 %add8.i., 10, !op_uniform !12, !res_uniform !12, !unaligned !12, !same !12
  %add34.i. = add i32 %mul33.i., %add.i., !op_varying !12, !res_vector !12, !unaligned !12, !consecutive !12
  %idxprom.i. = zext i32 %add34.i. to i64, !op_varying !12, !res_vector !12, !unaligned !12, !consecutive !12
  %35 = getelementptr i32, i32* %args, i64 %idxprom.i., !op_varying !12, !res_vector !12, !unaligned !12, !consecutive !12
  %pktPtrCast = bitcast i32* %35 to <4 x i32>*, !wfv_pkt_ptr_cast !12, !op_uniform !12, !res_vector !12, !unaligned !12, !consecutive !12
  %unpackW9 = extractelement <4 x i32> %conv32.i., i32 0, !wfv_pack_unpack !12
  %unpackW210 = extractelement <4 x i32> %conv32.i., i32 1, !wfv_pack_unpack !12
  %unpackW311 = extractelement <4 x i32> %conv32.i., i32 2, !wfv_pack_unpack !12
  %unpackW412 = extractelement <4 x i32> %conv32.i., i32 3, !wfv_pack_unpack !12
  %36 = getelementptr <4 x i32>, <4 x i32>* %pktPtrCast, i32 0, i32 0, !wfv_pack_unpack !12
  %37 = getelementptr <4 x i32>, <4 x i32>* %pktPtrCast, i32 0, i32 1, !wfv_pack_unpack !12
  %38 = getelementptr <4 x i32>, <4 x i32>* %pktPtrCast, i32 0, i32 2, !wfv_pack_unpack !12
  %39 = getelementptr <4 x i32>, <4 x i32>* %pktPtrCast, i32 0, i32 3, !wfv_pack_unpack !12
  %40 = extractelement <4 x i1> %loopMaskUpdate., i32 0, !op_uniform !12, !res_uniform !12
  br i1 %40, label %casc.exec0, label %casc.if1, !op_uniform !12

casc.exec0:                                       ; preds = %while.end.i.
  store i32 %unpackW9, i32* %36, align 4, !op_uniform !12, !res_uniform !12
  br label %casc.if1, !op_uniform !12

casc.if1:                                         ; preds = %while.end.i., %casc.exec0
  %41 = extractelement <4 x i1> %loopMaskUpdate., i32 1, !op_uniform !12, !res_uniform !12
  br i1 %41, label %casc.exec1, label %casc.if2, !op_uniform !12

casc.exec1:                                       ; preds = %casc.if1
  store i32 %unpackW210, i32* %37, align 4, !op_uniform !12, !res_uniform !12
  br label %casc.if2, !op_uniform !12

casc.if2:                                         ; preds = %casc.if1, %casc.exec1
  %42 = extractelement <4 x i1> %loopMaskUpdate., i32 2, !op_uniform !12, !res_uniform !12
  br i1 %42, label %casc.exec2, label %casc.if3, !op_uniform !12

casc.exec2:                                       ; preds = %casc.if2
  store i32 %unpackW311, i32* %38, align 4, !op_uniform !12, !res_uniform !12
  br label %casc.if3, !op_uniform !12

casc.if3:                                         ; preds = %casc.if2, %casc.exec2
  %43 = extractelement <4 x i1> %loopMaskUpdate., i32 3, !op_uniform !12, !res_uniform !12
  br i1 %43, label %casc.exec3, label %while.end.i..casc.end, !op_uniform !12

casc.exec3:                                       ; preds = %casc.if3
  store i32 %unpackW412, i32* %39, align 4, !op_uniform !12, !res_uniform !12
  br label %while.end.i..casc.end, !op_uniform !12

while.end.i..casc.end:                            ; preds = %casc.if3, %casc.exec3
  br label %"_ZZ4mainENK3$_0clEPiffff.exit.", !op_uniform !12, !res_uniform !12

"_ZZ4mainENK3$_0clEPiffff.exit.":                 ; preds = %while.end.i..casc.end
  ret void, !op_uniform !12, !res_uniform !12
}

; Function Attrs: nounwind readnone
declare void @wfvMetadataFn() #3

; Function Attrs: nounwind readnone
declare i32 @wfv_unpack(i32) #3

; Function Attrs: nounwind readnone
declare i32* @wfv_unpack_(i32*) #3

; Function Attrs: nounwind readnone
declare i32 @wfv_Wunpack(i32, i32) #3

; Function Attrs: nounwind readnone
declare i32* @wfv_Wunpack_(i32*, i32) #3

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
!9 = !{i32 0, i32 1024}
!10 = !{i32 0, i32 65535}
!11 = !{i32 1, i32 1025}
!12 = !{null}
