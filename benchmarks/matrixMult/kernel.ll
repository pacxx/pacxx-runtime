; ModuleID = 'kernel.ll'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-unknown-unknown"

@__cudart_i2opi_f = internal addrspace(4) global [6 x i32] [i32 1011060801, i32 -614296167, i32 -181084736, i32 -64530479, i32 1313084713, i32 -1560706194], align 4
@__cudart_i2opi_d = internal addrspace(4) global [18 x i64] [i64 7780917995555872008, i64 4397547296490951402, i64 8441921394348257659, i64 5712322887342352941, i64 7869616827067468215, i64 -1211730484530615009, i64 2303758334597371919, i64 -7168499653074671557, i64 4148332274289687028, i64 -1613291254968254911, i64 -1692731182770600828, i64 -135693905287338178, i64 452944820249399836, i64 -5249950069107600672, i64 -121206125134887583, i64 -2638381946312093631, i64 -277156292786332224, i64 -6703182060581546711], align 8

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

define ptx_kernel void @"_ZN5pacxx2v213genericKernelILm0EZ10calcNativePfS2_S2_mE12$_1209590108JS2_S2_S2_iEEEvT0_DpNSt3__111conditionalIXsr3std12is_referenceIT1_EE5valueENS5_20add_lvalue_referenceINS0_17generic_to_globalIS7_E4typeEE4typeESB_E4typeE"(i8 %callable.coerce, float addrspace(1)* %args, float addrspace(1)* %args1, float addrspace(1)* %args2, i32 %args3) {
  %1 = call spir_func i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1
  %2 = call spir_func i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1
  %3 = call spir_func i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1
  %4 = mul i32 %2, %1
  %5 = add i32 %4, %3
  %6 = call spir_func i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #1
  %7 = call spir_func i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #1
  %8 = call spir_func i32 @llvm.nvvm.read.ptx.sreg.tid.y() #1
  %9 = mul i32 %7, %6
  %10 = add i32 %9, %8
  %11 = icmp sgt i32 %args3, 0
  %12 = mul nsw i32 %10, %args3
  br i1 %11, label %.lr.ph.i, label %"_ZZ10calcNativePfS_S_mENK12$_1209590108clEPKfS2_S_i.exit"

.lr.ph.i:                                         ; preds = %.lr.ph.i, %0
  %i.08.i = phi i32 [ 0, %0 ], [ %28, %.lr.ph.i ]
  %val.07.i = phi float [ 0.000000e+00, %0 ], [ %27, %.lr.ph.i ]
  %13 = add nsw i32 %i.08.i, %12
  %14 = sext i32 %13 to i64
  %15 = getelementptr float, float addrspace(1)* %args, i64 %14
  %16 = ptrtoint float addrspace(1)* %15 to i64, !pacxx.addrspace !18
  %17 = inttoptr i64 %16 to float*
  %18 = mul nsw i32 %i.08.i, %args3
  %19 = add nsw i32 %18, %5
  %20 = sext i32 %19 to i64
  %21 = getelementptr float, float addrspace(1)* %args1, i64 %20
  %22 = ptrtoint float addrspace(1)* %21 to i64, !pacxx.addrspace !18
  %23 = inttoptr i64 %22 to float*
  %24 = load float, float* %17, align 4, !alias.scope !19, !noalias !22
  %25 = load float, float* %23, align 4, !alias.scope !25, !noalias !26
  %26 = fmul float %24, %25
  %27 = fadd float %val.07.i, %26
  %28 = add nsw i32 %i.08.i, 1
  %29 = icmp slt i32 %28, %args3
  br i1 %29, label %.lr.ph.i, label %"_ZZ10calcNativePfS_S_mENK12$_1209590108clEPKfS2_S_i.exit"

"_ZZ10calcNativePfS_S_mENK12$_1209590108clEPKfS2_S_i.exit": ; preds = %.lr.ph.i, %0
  %val.0.lcssa.i = phi float [ 0.000000e+00, %0 ], [ %27, %.lr.ph.i ]
  %30 = add nsw i32 %12, %5
  %31 = sext i32 %30 to i64
  %32 = getelementptr float, float addrspace(1)* %args2, i64 %31
  %33 = ptrtoint float addrspace(1)* %32 to i64, !pacxx.addrspace !18
  %34 = inttoptr i64 %33 to float*
  store float %val.0.lcssa.i, float* %34, align 4, !alias.scope !27, !noalias !28
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }

!llvm.ident = !{!0, !1, !0}
!nvvm.annotations = !{!2, !3, !4, !3, !5, !5, !5, !5, !6, !6, !5}
!opencl.kernels = !{!7}
!opencl.spir.version = !{!14}
!opencl.ocl.version = !{!14}
!opencl.enable.FP_CONTRACT = !{!15}
!opencl.used.optional.core.features = !{!15}
!opencl.used.extensions = !{!15}
!opencl.compiler.options = !{!15}
!pacxx.kernel = !{!16}
!pacxx.kernel._ZN5pacxx2v213genericKernelILm0EZ10calcNativePfS2_S2_mE12$_1209590108JS2_S2_S2_iEEEvT0_DpNSt3__111conditionalIXsr3std12is_referenceIT1_EE5valueENS5_20add_lvalue_referenceINS0_17generic_to_globalIS7_E4typeEE4typeESB_E4typeE = !{!17}
!nvvm.internalize.after.link = !{}
!nvvmir.version = !{!14}

!0 = !{!"clang version 3.8.0 (https://lklein14@bitbucket.org/mhaidl/clang.git 35a35447d041832b6e2e25acaf7c825860f8f407) (https://lklein14@bitbucket.org/lklein14/llvm.git 3a43138fba13912a93303daeeb802aa41cd21335)"}
!1 = !{!"clang version 3.8.0 (https://lklein14@bitbucket.org/mhaidl/clang.git 35a35447d041832b6e2e25acaf7c825860f8f407) (https://lklein14@bitbucket.org/lklein14/llvm.git 1038c73267508716cd4557f0233750e75c900f93)"}
!2 = !{void (i8, float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32)* @"_ZN5pacxx2v213genericKernelILm0EZ10calcNativePfS2_S2_mE12$_1209590108JS2_S2_S2_iEEEvT0_DpNSt3__111conditionalIXsr3std12is_referenceIT1_EE5valueENS5_20add_lvalue_referenceINS0_17generic_to_globalIS7_E4typeEE4typeESB_E4typeE", !"kernel", i32 1}
!3 = !{null, !"align", i32 8}
!4 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!5 = !{null, !"align", i32 16}
!6 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!7 = !{void (i8, float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32)* @"_ZN5pacxx2v213genericKernelILm0EZ10calcNativePfS2_S2_mE12$_1209590108JS2_S2_S2_iEEEvT0_DpNSt3__111conditionalIXsr3std12is_referenceIT1_EE5valueENS5_20add_lvalue_referenceINS0_17generic_to_globalIS7_E4typeEE4typeESB_E4typeE", !8, !9, !10, !11, !12, !13}
!8 = !{!"kernel_arg_addr_space", i32 0, i32 1, i32 1, i32 1, i32 0}
!9 = !{!"kernel_arg_type", !"class (lambda at Native.cpp:69:15)", !"std::conditional_t<std::is_reference<float*>::value, std::add_lvalue_reference_t<typename generic_to_global<float *>::type>, typename generic_to_global<float *>::type>", !"std::conditional_t<std::is_reference<float*>::value, std::add_lvalue_reference_t<typename generic_to_global<float *>::type>, typename generic_to_global<float *>::type>", !"std::conditional_t<std::is_reference<float*>::value, std::add_lvalue_reference_t<typename generic_to_global<float *>::type>, typename generic_to_global<float *>::type>", !"std::conditional_t<std::is_reference<int>::value, std::add_lvalue_reference_t<typename generic_to_global<int>::type>, typename generic_to_global<int>::type>"}
!10 = !{!"kernel_arg_name", !"callable", !"args", !"args", !"args", !"args"}
!11 = !{!"kernel_arg_access_qual", !"", !"", !"", !"", !""}
!12 = !{!"kernel_arg_base_type", !"class (lambda at Native.cpp:69:15)", !" float*", !" float*", !" float*", !"int"}
!13 = !{!"kernel_arg_type_qual", !"", !"", !"", !"", !""}
!14 = !{i32 1, i32 2}
!15 = !{}
!16 = !{void (i8, float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32)* @"_ZN5pacxx2v213genericKernelILm0EZ10calcNativePfS2_S2_mE12$_1209590108JS2_S2_S2_iEEEvT0_DpNSt3__111conditionalIXsr3std12is_referenceIT1_EE5valueENS5_20add_lvalue_referenceINS0_17generic_to_globalIS7_E4typeEE4typeESB_E4typeE"}
!17 = !{i32 -1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 -1, i32 0, i32 0}
!18 = !{i64 0}
!19 = !{!20}
!20 = distinct !{!20, !21, !"_ZZ10calcNativePfS_S_mENK12$_1209590108clEPKfS2_S_i: %a"}
!21 = distinct !{!21, !"_ZZ10calcNativePfS_S_mENK12$_1209590108clEPKfS2_S_i"}
!22 = !{!23, !24}
!23 = distinct !{!23, !21, !"_ZZ10calcNativePfS_S_mENK12$_1209590108clEPKfS2_S_i: %b"}
!24 = distinct !{!24, !21, !"_ZZ10calcNativePfS_S_mENK12$_1209590108clEPKfS2_S_i: %c"}
!25 = !{!23}
!26 = !{!20, !24}
!27 = !{!24}
!28 = !{!20, !23}
