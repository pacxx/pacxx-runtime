; ModuleID = './tmp-161219-1130-886hpW/spir.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir64-unknown-unknown"

define spir_kernel void @"_ZN5pacxx2v213genericKernelILm0EZ4mainE12$_4028513607JPfS3_S3_iEEEvT0_DpNSt3__111conditionalIXsr3std12is_referenceIT1_EE5valueENS5_20add_lvalue_referenceINS0_17generic_to_globalIS7_E4typeEE4typeESB_E4typeE"(i8 %callable.coerce, float addrspace(1)* %args, float addrspace(1)* %args1, float addrspace(1)* %args2, i32 %args3) {
  %1 = tail call spir_func i32 @_Z13get_global_idj(i32 0)
  %2 = icmp ult i32 %1, %args3
  br i1 %2, label %3, label %"_ZZ4mainENK12$_4028513607clEPKfS1_Pfj.exit"

; <label>:3                                       ; preds = %0
  %4 = tail call i32 @_Z13get_global_idj(i32 0)
  %5 = sext i32 %4 to i64
  %6 = getelementptr float, float addrspace(1)* %args, i64 %5
  %7 = ptrtoint float addrspace(1)* %6 to i64, !pacxx.addrspace !11
  %8 = inttoptr i64 %7 to float*
  %9 = addrspacecast float addrspace(1)* %6 to float*
  %10 = getelementptr float, float addrspace(1)* %args1, i64 %5
  %11 = ptrtoint float addrspace(1)* %10 to i64, !pacxx.addrspace !11
  %12 = inttoptr i64 %11 to float*
  %13 = addrspacecast float addrspace(1)* %10 to float*
  %14 = load float, float* %8, align 4
  %15 = load float, float* %12, align 4
  %16 = fadd float %14, %15
  %17 = getelementptr float, float addrspace(1)* %args2, i64 %5
  %18 = ptrtoint float addrspace(1)* %17 to i64, !pacxx.addrspace !11
  %19 = inttoptr i64 %18 to float*
  %20 = addrspacecast float addrspace(1)* %17 to float*
  store float %16, float* %19, align 4
  br label %"_ZZ4mainENK12$_4028513607clEPKfS1_Pfj.exit"

"_ZZ4mainENK12$_4028513607clEPKfS1_Pfj.exit":     ; preds = %0, %3
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
!llvm.ident = !{!10, !10}

!0 = !{void (i8, float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32)* @"_ZN5pacxx2v213genericKernelILm0EZ4mainE12$_4028513607JPfS3_S3_iEEEvT0_DpNSt3__111conditionalIXsr3std12is_referenceIT1_EE5valueENS5_20add_lvalue_referenceINS0_17generic_to_globalIS7_E4typeEE4typeESB_E4typeE", !"kernel", i32 1}
!1 = !{void (i8, float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32)* @"_ZN5pacxx2v213genericKernelILm0EZ4mainE12$_4028513607JPfS3_S3_iEEEvT0_DpNSt3__111conditionalIXsr3std12is_referenceIT1_EE5valueENS5_20add_lvalue_referenceINS0_17generic_to_globalIS7_E4typeEE4typeESB_E4typeE", !2, !3, !4, !5, !6, !7}
!2 = !{!"kernel_arg_addr_space", i32 0, i32 1, i32 1, i32 1, i32 0}
!3 = !{!"kernel_arg_type", !"class (lambda at vectorAdd.cpp:60:15)", !"std::conditional_t<std::is_reference<float*>::value, std::add_lvalue_reference_t<typename generic_to_global<float *>::type>, typename generic_to_global<float *>::type>", !"std::conditional_t<std::is_reference<float*>::value, std::add_lvalue_reference_t<typename generic_to_global<float *>::type>, typename generic_to_global<float *>::type>", !"std::conditional_t<std::is_reference<float*>::value, std::add_lvalue_reference_t<typename generic_to_global<float *>::type>, typename generic_to_global<float *>::type>", !"std::conditional_t<std::is_reference<int>::value, std::add_lvalue_reference_t<typename generic_to_global<int>::type>, typename generic_to_global<int>::type>"}
!4 = !{!"kernel_arg_name", !"callable", !"args", !"args", !"args", !"args"}
!5 = !{!"kernel_arg_access_qual", !"", !"", !"", !"", !""}
!6 = !{!"kernel_arg_base_type", !"class (lambda at vectorAdd.cpp:60:15)", !" float*", !" float*", !" float*", !"int"}
!7 = !{!"kernel_arg_type_qual", !"", !"", !"", !"", !""}
!8 = !{i32 1, i32 2}
!9 = !{}
!10 = !{!"clang version 3.8.0 (https://lklein14@bitbucket.org/mhaidl/clang.git 35a35447d041832b6e2e25acaf7c825860f8f407) (https://lklein14@bitbucket.org/lklein14/llvm.git 75875af22bb6eb137144abef917003682624712f)"}
!11 = !{i64 0}
