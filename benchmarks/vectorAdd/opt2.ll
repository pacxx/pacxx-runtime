; ModuleID = './tmp-161208-1147-jLcTHU/pacxx.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "pacxx-unknown-unknown"

%struct.Thread = type { %struct._idx, %struct._idx }
%struct._idx = type { i32, i32, i32 }

; Function Attrs: noinline
define spir_kernel void @"_ZN5pacxx2v213genericKernelILm0EZ4mainE12$_3675094202JPfS3_S3_iEEEvT0_DpNSt3__111conditionalIXsr3std12is_referenceIT1_EE5valueENS5_20add_lvalue_referenceINS0_17generic_to_globalIS7_E4typeEE4typeESB_E4typeE"(i8 %callable.coerce, float addrspace(1)* nocapture readonly %args, float addrspace(1)* nocapture readonly %args1, float addrspace(1)* nocapture %args2, i32 %args3) #0 {
  %1 = tail call i32 @_Z12get_local_idj(i32 0)
  %2 = tail call i32 @_Z12get_local_idj(i32 1)
  %3 = tail call i32 @_Z12get_local_idj(i32 2)
  %4 = tail call i32 @_Z13get_global_idj(i32 0)
  %5 = tail call i32 @_Z13get_global_idj(i32 1)
  %6 = tail call i32 @_Z13get_global_idj(i32 2)
  %7 = icmp slt i32 %4, %args3
  br i1 %7, label %8, label %"_ZZ4mainENK12$_3675094202clEPKfS1_Pfi.exit"

; <label>:8                                       ; preds = %0
  %9 = sext i32 %4 to i64
  %10 = getelementptr inbounds float, float addrspace(1)* %args, i64 %9
  %11 = addrspacecast float addrspace(1)* %10 to float*
  %12 = load float, float* %11, align 4
  %13 = getelementptr inbounds float, float addrspace(1)* %args1, i64 %9
  %14 = addrspacecast float addrspace(1)* %13 to float*
  %15 = load float, float* %14, align 4
  %16 = fadd float %12, %15
  %17 = getelementptr inbounds float, float addrspace(1)* %args2, i64 %9
  %18 = addrspacecast float addrspace(1)* %17 to float*
  store float %16, float* %18, align 4
  br label %"_ZZ4mainENK12$_3675094202clEPKfS1_Pfi.exit"

"_ZZ4mainENK12$_3675094202clEPKfS1_Pfi.exit":     ; preds = %0, %8
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
!llvm.ident = !{!10, !10}

!0 = !{void (i8, float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32)* @"_ZN5pacxx2v213genericKernelILm0EZ4mainE12$_3675094202JPfS3_S3_iEEEvT0_DpNSt3__111conditionalIXsr3std12is_referenceIT1_EE5valueENS5_20add_lvalue_referenceINS0_17generic_to_globalIS7_E4typeEE4typeESB_E4typeE", !"kernel", i32 1}
!1 = !{void (i8, float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32)* @"_ZN5pacxx2v213genericKernelILm0EZ4mainE12$_3675094202JPfS3_S3_iEEEvT0_DpNSt3__111conditionalIXsr3std12is_referenceIT1_EE5valueENS5_20add_lvalue_referenceINS0_17generic_to_globalIS7_E4typeEE4typeESB_E4typeE", !2, !3, !4, !5, !6, !7}
!2 = !{!"kernel_arg_addr_space", i32 0, i32 1, i32 1, i32 1, i32 0}
!3 = !{!"kernel_arg_type", !"class (lambda at Native.cpp:55:15)", !"std::conditional_t<std::is_reference<float*>::value, std::add_lvalue_reference_t<typename generic_to_global<float *>::type>, typename generic_to_global<float *>::type>", !"std::conditional_t<std::is_reference<float*>::value, std::add_lvalue_reference_t<typename generic_to_global<float *>::type>, typename generic_to_global<float *>::type>", !"std::conditional_t<std::is_reference<float*>::value, std::add_lvalue_reference_t<typename generic_to_global<float *>::type>, typename generic_to_global<float *>::type>", !"std::conditional_t<std::is_reference<int>::value, std::add_lvalue_reference_t<typename generic_to_global<int>::type>, typename generic_to_global<int>::type>"}
!4 = !{!"kernel_arg_name", !"callable", !"args", !"args", !"args", !"args"}
!5 = !{!"kernel_arg_access_qual", !"", !"", !"", !"", !""}
!6 = !{!"kernel_arg_base_type", !"class (lambda at Native.cpp:55:15)", !" float*", !" float*", !" float*", !"int"}
!7 = !{!"kernel_arg_type_qual", !"", !"", !"", !"", !""}
!8 = !{i32 1, i32 2}
!9 = !{}
!10 = !{!"clang version 3.8.0 (https://lklein14@bitbucket.org/mhaidl/clang.git 35a35447d041832b6e2e25acaf7c825860f8f407) (https://lklein14@bitbucket.org/lklein14/llvm.git 1038c73267508716cd4557f0233750e75c900f93)"}
