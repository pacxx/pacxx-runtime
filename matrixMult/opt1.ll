; ModuleID = './tmp-161208-1137-CTvR9u/pacxx.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "pacxx-unknown-unknown"

%struct.Thread = type { %struct._idx, %struct._idx }
%struct._idx = type { i32, i32, i32 }

; Function Attrs: noinline
define spir_kernel void @"_ZN5pacxx2v213genericKernelILm0EZ10calcNativePfS2_S2_mE12$_1209590108JS2_S2_S2_iEEEvT0_DpNSt3__111conditionalIXsr3std12is_referenceIT1_EE5valueENS5_20add_lvalue_referenceINS0_17generic_to_globalIS7_E4typeEE4typeESB_E4typeE"(i8 %callable.coerce, float addrspace(1)* nocapture readonly %args, float addrspace(1)* nocapture readonly %args1, float addrspace(1)* nocapture %args2, i32 %args3) #0 {
  %1 = tail call i32 @_Z12get_local_idj(i32 0), !noalias !12
  %2 = tail call i32 @_Z12get_local_idj(i32 1), !noalias !12
  %3 = tail call i32 @_Z12get_local_idj(i32 2), !noalias !12
  %4 = tail call i32 @_Z13get_global_idj(i32 0), !noalias !12
  %5 = tail call i32 @_Z13get_global_idj(i32 1), !noalias !12
  %6 = tail call i32 @_Z13get_global_idj(i32 2), !noalias !12
  %7 = tail call i32 @_Z12get_local_idj(i32 0), !noalias !12
  %8 = tail call i32 @_Z12get_local_idj(i32 1), !noalias !12
  %9 = tail call i32 @_Z12get_local_idj(i32 2), !noalias !12
  %10 = tail call i32 @_Z13get_global_idj(i32 0), !noalias !12
  %11 = tail call i32 @_Z13get_global_idj(i32 1), !noalias !12
  %12 = tail call i32 @_Z13get_global_idj(i32 2), !noalias !12
  %13 = icmp sgt i32 %args3, 0
  %14 = mul nsw i32 %11, %args3
  br i1 %13, label %.lr.ph.i, label %"_ZZ10calcNativePfS_S_mENK12$_1209590108clEPKfS2_S_i.exit"

.lr.ph.i:                                         ; preds = %.lr.ph.i, %0
  %val.0.i = phi float [ 0.000000e+00, %0 ], [ %27, %.lr.ph.i ]
  %i.0.i = phi i32 [ 0, %0 ], [ %28, %.lr.ph.i ]
  %15 = add nsw i32 %i.0.i, %14
  %16 = sext i32 %15 to i64
  %17 = getelementptr inbounds float, float addrspace(1)* %args, i64 %16
  %18 = addrspacecast float addrspace(1)* %17 to float*
  %19 = load float, float* %18, align 4, !alias.scope !17, !noalias !18
  %20 = mul nsw i32 %i.0.i, %args3
  %21 = add nsw i32 %20, %4
  %22 = sext i32 %21 to i64
  %23 = getelementptr inbounds float, float addrspace(1)* %args1, i64 %22
  %24 = addrspacecast float addrspace(1)* %23 to float*
  %25 = load float, float* %24, align 4, !alias.scope !19, !noalias !20
  %26 = fmul float %19, %25
  %27 = fadd float %val.0.i, %26
  %28 = add nuw nsw i32 %i.0.i, 1
  %29 = icmp slt i32 %28, %args3
  br i1 %29, label %.lr.ph.i, label %"_ZZ10calcNativePfS_S_mENK12$_1209590108clEPKfS2_S_i.exit"

"_ZZ10calcNativePfS_S_mENK12$_1209590108clEPKfS2_S_i.exit": ; preds = %.lr.ph.i, %0
  %val.1.i = phi float [ 0.000000e+00, %0 ], [ %27, %.lr.ph.i ]
  %30 = add nsw i32 %14, %4
  %31 = sext i32 %30 to i64
  %32 = getelementptr inbounds float, float addrspace(1)* %args2, i64 %31
  %33 = addrspacecast float addrspace(1)* %32 to float*
  store float %val.1.i, float* %33, align 4, !alias.scope !21, !noalias !22
  ret void
}

define i32 @_ZN6native5index1xILi0EEEjv() #1 {
  %1 = tail call i32 @_Z12get_local_idj(i32 0)
  ret i32 %1
}

declare i32 @_Z12get_local_idj(i32) #1

define i32 @_ZN6native5index1yILi0EEEjv() #1 {
  %1 = tail call i32 @_Z12get_local_idj(i32 1)
  ret i32 %1
}

define i32 @_ZN6native5index1zILi0EEEjv() #1 {
  %1 = tail call i32 @_Z12get_local_idj(i32 2)
  ret i32 %1
}

define i32 @_ZN6native5index1xILi2EEEjv() #1 {
  %1 = tail call i32 @_Z13get_global_idj(i32 0)
  ret i32 %1
}

declare i32 @_Z13get_global_idj(i32) #1

define i32 @_ZN6native5index1yILi2EEEjv() #1 {
  %1 = tail call i32 @_Z13get_global_idj(i32 1)
  ret i32 %1
}

define i32 @_ZN6native5index1zILi2EEEjv() #1 {
  %1 = tail call i32 @_Z13get_global_idj(i32 2)
  ret i32 %1
}

define %struct.Thread @_ZN6Thread3getEv() #1 align 2 {
  %1 = tail call i32 @_Z12get_local_idj(i32 0)
  %2 = tail call i32 @_Z12get_local_idj(i32 1)
  %3 = tail call i32 @_Z12get_local_idj(i32 2)
  %4 = tail call i32 @_Z13get_global_idj(i32 0)
  %5 = tail call i32 @_Z13get_global_idj(i32 1)
  %6 = tail call i32 @_Z13get_global_idj(i32 2)
  %7 = insertvalue %struct.Thread undef, i32 %1, 0, 0
  %8 = insertvalue %struct.Thread %7, i32 %2, 0, 1
  %9 = insertvalue %struct.Thread %8, i32 %3, 0, 2
  %10 = insertvalue %struct.Thread %9, i32 %4, 1, 0
  %11 = insertvalue %struct.Thread %10, i32 %5, 1, 1
  %12 = insertvalue %struct.Thread %11, i32 %6, 1, 2
  ret %struct.Thread %12
}

attributes #0 = { noinline "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

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
!17 = !{!13}
!18 = !{!15, !16}
!19 = !{!15}
!20 = !{!13, !16}
!21 = !{!16}
!22 = !{!13, !15}
