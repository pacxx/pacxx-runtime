; ModuleID = './tmp-161208-1137-CTvR9u/nvvm.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-unknown-unknown"

declare i32 @vprintf(i8* nocapture, i8*)

define void @__printf(i8 addrspace(4)* %ptr, i8* %val) {
entry:
  %0 = addrspacecast i8 addrspace(4)* %ptr to i8*
  %call = call i32 @vprintf(i8* %0, i8* %val)
  ret void
}

define ptx_kernel void @"_ZN5pacxx2v213genericKernelILm0EZ10calcNativePfS2_S2_mE12$_1209590108JS2_S2_S2_iEEEvT0_DpNSt3__111conditionalIXsr3std12is_referenceIT1_EE5valueENS5_20add_lvalue_referenceINS0_17generic_to_globalIS7_E4typeEE4typeESB_E4typeE"(i8 %callable.coerce, float addrspace(1)* %args, float addrspace(1)* %args1, float addrspace(1)* %args2, i32 %args3) {
_Z13get_global_idj.exit:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #1
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #1
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #1
  %3 = mul i32 %0, %1
  %4 = add i32 %3, %2
  %5 = icmp sgt i32 %args3, 0
  %6 = mul nsw i32 %4, %args3
  br i1 %5, label %.lr.ph.i, label %"_ZZ10calcNativePfS_S_mENK12$_1209590108clEPKfS2_S_i.exit"

.lr.ph.i:                                         ; preds = %_Z13get_global_idj.exit, %.lr.ph.i
  %.01 = phi i32 [ undef, %_Z13get_global_idj.exit ], [ %11, %.lr.ph.i ]
  %val.0.i = phi float [ 0.000000e+00, %_Z13get_global_idj.exit ], [ %26, %.lr.ph.i ]
  %i.0.i = phi i32 [ 0, %_Z13get_global_idj.exit ], [ %27, %.lr.ph.i ]
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1
  %8 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1
  %9 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1
  %10 = mul i32 %7, %8
  %11 = add i32 %10, %9
  %12 = add nsw i32 %i.0.i, %6
  %13 = sext i32 %12 to i64
  %14 = getelementptr float, float addrspace(1)* %args, i64 %13
  %15 = ptrtoint float addrspace(1)* %14 to i64, !pacxx.addrspace !18
  %16 = inttoptr i64 %15 to float*
  %17 = mul nsw i32 %i.0.i, %args3
  %18 = add nsw i32 %17, %11
  %19 = sext i32 %18 to i64
  %20 = getelementptr float, float addrspace(1)* %args1, i64 %19
  %21 = ptrtoint float addrspace(1)* %20 to i64, !pacxx.addrspace !18
  %22 = inttoptr i64 %21 to float*
  %23 = load float, float* %16, align 4, !alias.scope !19, !noalias !22
  %24 = load float, float* %22, align 4, !alias.scope !25, !noalias !26
  %25 = fmul float %23, %24
  %26 = fadd float %val.0.i, %25
  %27 = add nuw nsw i32 %i.0.i, 1
  %28 = icmp slt i32 %27, %args3
  br i1 %28, label %.lr.ph.i, label %"_ZZ10calcNativePfS_S_mENK12$_1209590108clEPKfS2_S_i.exit"

"_ZZ10calcNativePfS_S_mENK12$_1209590108clEPKfS2_S_i.exit": ; preds = %_Z13get_global_idj.exit, %.lr.ph.i
  %val.1.i = phi float [ 0.000000e+00, %_Z13get_global_idj.exit ], [ %26, %.lr.ph.i ]
  %29 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1
  %30 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1
  %31 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1
  %32 = mul i32 %29, %30
  %33 = add i32 %32, %31
  %34 = add nsw i32 %6, %33
  %35 = sext i32 %34 to i64
  %36 = getelementptr float, float addrspace(1)* %args2, i64 %35
  %37 = ptrtoint float addrspace(1)* %36 to i64, !pacxx.addrspace !18
  %38 = inttoptr i64 %37 to float*
  store float %val.1.i, float* %38, align 4, !alias.scope !27, !noalias !28
  ret void
}

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

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }

!nvvm.annotations = !{!0, !1, !2, !1, !3, !3, !3, !3, !4, !4, !3}
!opencl.kernels = !{!5}
!opencl.spir.version = !{!12}
!opencl.ocl.version = !{!12}
!opencl.enable.FP_CONTRACT = !{!13}
!opencl.used.optional.core.features = !{!13}
!opencl.used.extensions = !{!13}
!opencl.compiler.options = !{!13}
!llvm.ident = !{!14, !15, !15}
!nvvm.internalize.after.link = !{}
!nvvmir.version = !{!12}
!pacxx.kernel = !{!16}
!pacxx.kernel._ZN5pacxx2v213genericKernelILm0EZ10calcNativePfS2_S2_mE12$_1209590108JS2_S2_S2_iEEEvT0_DpNSt3__111conditionalIXsr3std12is_referenceIT1_EE5valueENS5_20add_lvalue_referenceINS0_17generic_to_globalIS7_E4typeEE4typeESB_E4typeE = !{!17}

!0 = !{void (i8, float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32)* @"_ZN5pacxx2v213genericKernelILm0EZ10calcNativePfS2_S2_mE12$_1209590108JS2_S2_S2_iEEEvT0_DpNSt3__111conditionalIXsr3std12is_referenceIT1_EE5valueENS5_20add_lvalue_referenceINS0_17generic_to_globalIS7_E4typeEE4typeESB_E4typeE", !"kernel", i32 1}
!1 = !{null, !"align", i32 8}
!2 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!3 = !{null, !"align", i32 16}
!4 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!5 = !{void (i8, float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32)* @"_ZN5pacxx2v213genericKernelILm0EZ10calcNativePfS2_S2_mE12$_1209590108JS2_S2_S2_iEEEvT0_DpNSt3__111conditionalIXsr3std12is_referenceIT1_EE5valueENS5_20add_lvalue_referenceINS0_17generic_to_globalIS7_E4typeEE4typeESB_E4typeE", !6, !7, !8, !9, !10, !11}
!6 = !{!"kernel_arg_addr_space", i32 0, i32 1, i32 1, i32 1, i32 0}
!7 = !{!"kernel_arg_type", !"class (lambda at Native.cpp:69:15)", !"std::conditional_t<std::is_reference<float*>::value, std::add_lvalue_reference_t<typename generic_to_global<float *>::type>, typename generic_to_global<float *>::type>", !"std::conditional_t<std::is_reference<float*>::value, std::add_lvalue_reference_t<typename generic_to_global<float *>::type>, typename generic_to_global<float *>::type>", !"std::conditional_t<std::is_reference<float*>::value, std::add_lvalue_reference_t<typename generic_to_global<float *>::type>, typename generic_to_global<float *>::type>", !"std::conditional_t<std::is_reference<int>::value, std::add_lvalue_reference_t<typename generic_to_global<int>::type>, typename generic_to_global<int>::type>"}
!8 = !{!"kernel_arg_name", !"callable", !"args", !"args", !"args", !"args"}
!9 = !{!"kernel_arg_access_qual", !"", !"", !"", !"", !""}
!10 = !{!"kernel_arg_base_type", !"class (lambda at Native.cpp:69:15)", !" float*", !" float*", !" float*", !"int"}
!11 = !{!"kernel_arg_type_qual", !"", !"", !"", !"", !""}
!12 = !{i32 1, i32 2}
!13 = !{}
!14 = !{!"clang version 3.8.0 (https://lklein14@bitbucket.org/mhaidl/clang.git 35a35447d041832b6e2e25acaf7c825860f8f407) (https://lklein14@bitbucket.org/lklein14/llvm.git 1038c73267508716cd4557f0233750e75c900f93)"}
!15 = !{!"clang version 3.8.0 (https://lklein14@bitbucket.org/mhaidl/clang.git 35a35447d041832b6e2e25acaf7c825860f8f407) (https://lklein14@bitbucket.org/lklein14/llvm.git 3a43138fba13912a93303daeeb802aa41cd21335)"}
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
