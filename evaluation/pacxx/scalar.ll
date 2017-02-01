
; Function Attrs: noinline nounwind uwtable
define void @"_ZN5pacxx2v213genericKernelILm0EZ4mainE3$_0JPiffffEEEvT0_DpT1_"(i8 %callable.coerce, i32* nocapture %args, float %args1, float %args3, float %args5, float %args7) #2 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
IdCalc:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #4, !range !7
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #4, !range !8
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #4, !range !9
  %mul6.i = mul nuw nsw i32 %2, %1
  %add8.i = add nuw nsw i32 %mul6.i, %0
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #4, !range !7
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #4, !range !8
  %5 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #4, !range !9
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
