; ModuleID = './tmp-161208-1137-CTvR9u/spir.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir64-unknown-unknown"

define spir_kernel void @"_ZN5pacxx2v213genericKernelILm0EZ10calcNativePfS2_S2_mE12$_1209590108JS2_S2_S2_iEEEvT0_DpNSt3__111conditionalIXsr3std12is_referenceIT1_EE5valueENS5_20add_lvalue_referenceINS0_17generic_to_globalIS7_E4typeEE4typeESB_E4typeE"(i8 %callable.coerce, float addrspace(1)* %args, float addrspace(1)* %args1, float addrspace(1)* %args2, i32 %args3) {
  %1 = tail call spir_func i32 @_Z13get_global_idj(i32 0), !noalias !12
  %2 = tail call spir_func i32 @_Z13get_global_idj(i32 1), !noalias !12
  %3 = icmp sgt i32 %args3, 0
  %4 = mul nsw i32 %2, %args3
  br i1 %3, label %.lr.ph.i, label %"_ZZ10calcNativePfS_S_mENK12$_1209590108clEPKfS2_S_i.exit"

.lr.ph.i:                                         ; preds = %.lr.ph.i, %0
  %val.0.i = phi float [ 0.000000e+00, %0 ], [ %22, %.lr.ph.i ]
  %i.0.i = phi i32 [ 0, %0 ], [ %23, %.lr.ph.i ]
  %5 = tail call i32 @_Z13get_global_idj(i32 0)
  %6 = add nsw i32 %i.0.i, %4
  %7 = sext i32 %6 to i64
  %8 = getelementptr float, float addrspace(1)* %args, i64 %7
  %9 = ptrtoint float addrspace(1)* %8 to i64, !pacxx.addrspace !17
  %10 = inttoptr i64 %9 to float*
  %11 = addrspacecast float addrspace(1)* %8 to float*
  %12 = mul nsw i32 %i.0.i, %args3
  %13 = add nsw i32 %12, %5
  %14 = sext i32 %13 to i64
  %15 = getelementptr float, float addrspace(1)* %args1, i64 %14
  %16 = ptrtoint float addrspace(1)* %15 to i64, !pacxx.addrspace !17
  %17 = inttoptr i64 %16 to float*
  %18 = addrspacecast float addrspace(1)* %15 to float*
  %19 = load float, float* %10, align 4, !alias.scope !18, !noalias !19
  %20 = load float, float* %17, align 4, !alias.scope !20, !noalias !21
  %21 = fmul float %19, %20
  %22 = fadd float %val.0.i, %21
  %23 = add nuw nsw i32 %i.0.i, 1
  %24 = icmp slt i32 %23, %args3
  br i1 %24, label %.lr.ph.i, label %"_ZZ10calcNativePfS_S_mENK12$_1209590108clEPKfS2_S_i.exit"

"_ZZ10calcNativePfS_S_mENK12$_1209590108clEPKfS2_S_i.exit": ; preds = %.lr.ph.i, %0
  %val.1.i = phi float [ 0.000000e+00, %0 ], [ %22, %.lr.ph.i ]
  %25 = tail call i32 @_Z13get_global_idj(i32 0)
  %26 = add nsw i32 %4, %25
  %27 = sext i32 %26 to i64
  %28 = getelementptr float, float addrspace(1)* %args2, i64 %27
  %29 = ptrtoint float addrspace(1)* %28 to i64, !pacxx.addrspace !17
  %30 = inttoptr i64 %29 to float*
  %31 = addrspacecast float addrspace(1)* %28 to float*
  store float %val.1.i, float* %30, align 4, !alias.scope !22, !noalias !23
  ret void
}

; Function Attrs: nounwind readnone
declare spir_func i32 @_Z12get_local_idj(i32) #0

; Function Attrs: nounwind readnone
declare spir_func i32 @_Z13get_global_idj(i32) #0

attributes #0 = { nounwind readnone }

!nvvm.annotations = !{!0}
!opencl.kernels = !{!1}
!opencl.spir.version = !{!8}
!opencl.ocl.version = !{!8}
!opencl.enable.FP_CONTRACT = !{!9}
!opencl.used.optional.core.features = !{!9}
!opencl.used.extensions = !{!9}
!opencl.compiler.options = !{!9}
!llvm.ident = !{!10, !11}

!0 = !{void (i8, float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32)* @"_ZN5pacxx2v213genericKernelILm0EZ10calcNativePfS2_S2_mE12$_1209590108JS2_S2_S2_iEEEvT0_DpNSt3__111conditionalIXsr3std12is_referenceIT1_EE5valueENS5_20add_lvalue_referenceINS0_17generic_to_globalIS7_E4typeEE4typeESB_E4typeE", !"kernel", i32 1}
!1 = !{void (i8, float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32)* @"_ZN5pacxx2v213genericKernelILm0EZ10calcNativePfS2_S2_mE12$_1209590108JS2_S2_S2_iEEEvT0_DpNSt3__111conditionalIXsr3std12is_referenceIT1_EE5valueENS5_20add_lvalue_referenceINS0_17generic_to_globalIS7_E4typeEE4typeESB_E4typeE", !2, !3, !4, !5, !6, !7}
!2 = !{!"kernel_arg_addr_space", i32 0, i32 1, i32 1, i32 1, i32 0}
!3 = !{!"kernel_arg_type", !"class (lambda at Native.cpp:69:15)", !"std::conditional_t<std::is_reference<float*>::value, std::add_lvalue_reference_t<typename generic_to_global<float *>::type>, typename generic_to_global<float *>::type>", !"std::conditional_t<std::is_reference<float*>::value, std::add_lvalue_reference_t<typename generic_to_global<float *>::type>, typename generic_to_global<float *>::type>", !"std::conditional_t<std::is_reference<float*>::value, std::add_lvalue_reference_t<typename generic_to_global<float *>::type>, typename generic_to_global<float *>::type>", !"std::conditional_t<std::is_reference<int>::value, std::add_lvalue_reference_t<typename generic_to_global<int>::type>, typename generic_to_global<int>::type>"}
!4 = !{!"kernel_arg_name", !"callable", !"args", !"args", !"args", !"args"}
!5 = !{!"kernel_arg_access_qual", !"", !"", !"", !"", !""}
!6 = !{!"kernel_arg_base_type", !"class (lambda at Native.cpp:69:15)", !" float*", !" float*", !" float*", !"int"}
!7 = !{!"kernel_arg_type_qual", !"", !"", !"", !"", !""}
!8 = !{i32 1, i32 2}
!9 = !{}
!10 = !{!"clang version 3.8.0 (https://lklein14@bitbucket.org/mhaidl/clang.git 35a35447d041832b6e2e25acaf7c825860f8f407) (https://lklein14@bitbucket.org/lklein14/llvm.git 1038c73267508716cd4557f0233750e75c900f93)"}
!11 = !{!"clang version 3.8.0 (https://lklein14@bitbucket.org/mhaidl/clang.git 35a35447d041832b6e2e25acaf7c825860f8f407) (https://lklein14@bitbucket.org/lklein14/llvm.git 3a43138fba13912a93303daeeb802aa41cd21335)"}
!12 = !{!13, !15, !16}
!13 = distinct !{!13, !14, !"_ZZ10calcNativePfS_S_mENK12$_1209590108clEPKfS2_S_i: %a"}
!14 = distinct !{!14, !"_ZZ10calcNativePfS_S_mENK12$_1209590108clEPKfS2_S_i"}
!15 = distinct !{!15, !14, !"_ZZ10calcNativePfS_S_mENK12$_1209590108clEPKfS2_S_i: %b"}
!16 = distinct !{!16, !14, !"_ZZ10calcNativePfS_S_mENK12$_1209590108clEPKfS2_S_i: %c"}
!17 = !{i64 0}
!18 = !{!13}
!19 = !{!15, !16}
!20 = !{!15}
!21 = !{!13, !16}
!22 = !{!16}
!23 = !{!13, !15}
