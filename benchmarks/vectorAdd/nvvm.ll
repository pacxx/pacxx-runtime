; ModuleID = './tmp-161208-1147-jLcTHU/nvvm.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-unknown-unknown"

declare i32 @vprintf(i8* nocapture, i8*)

define void @__printf(i8 addrspace(4)* %ptr, i8* %val) {
entry:
  %0 = addrspacecast i8 addrspace(4)* %ptr to i8*
  %call = call i32 @vprintf(i8* %0, i8* %val)
  ret void
}

define ptx_kernel void @"_ZN5pacxx2v213genericKernelILm0EZ4mainE12$_3675094202JPfS3_S3_iEEEvT0_DpNSt3__111conditionalIXsr3std12is_referenceIT1_EE5valueENS5_20add_lvalue_referenceINS0_17generic_to_globalIS7_E4typeEE4typeESB_E4typeE"(i8 %callable.coerce, float addrspace(1)* %args, float addrspace(1)* %args1, float addrspace(1)* %args2, i32 %args3) {
_Z13get_global_idj.exit:
  %0 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1
  %1 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1
  %3 = mul i32 %0, %1
  %4 = add i32 %3, %2
  %5 = icmp slt i32 %4, %args3
  br i1 %5, label %_Z13get_global_idj.exit4, label %"_ZZ4mainENK12$_3675094202clEPKfS1_Pfi.exit"

_Z13get_global_idj.exit4:                         ; preds = %_Z13get_global_idj.exit
  %6 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1
  %7 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1
  %8 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1
  %9 = mul i32 %6, %7
  %10 = add i32 %9, %8
  %11 = sext i32 %10 to i64
  %12 = getelementptr float, float addrspace(1)* %args, i64 %11
  %13 = ptrtoint float addrspace(1)* %12 to i64, !pacxx.addrspace !17
  %14 = inttoptr i64 %13 to float*
  %15 = getelementptr float, float addrspace(1)* %args1, i64 %11
  %16 = ptrtoint float addrspace(1)* %15 to i64, !pacxx.addrspace !17
  %17 = inttoptr i64 %16 to float*
  %18 = load float, float* %14, align 4
  %19 = load float, float* %17, align 4
  %20 = fadd float %18, %19
  %21 = getelementptr float, float addrspace(1)* %args2, i64 %11
  %22 = ptrtoint float addrspace(1)* %21 to i64, !pacxx.addrspace !17
  %23 = inttoptr i64 %22 to float*
  store float %20, float* %23, align 4
  br label %"_ZZ4mainENK12$_3675094202clEPKfS1_Pfi.exit"

"_ZZ4mainENK12$_3675094202clEPKfS1_Pfi.exit":     ; preds = %_Z13get_global_idj.exit, %_Z13get_global_idj.exit4
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

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
!llvm.ident = !{!14, !14, !14}
!nvvm.internalize.after.link = !{}
!nvvmir.version = !{!12}
!pacxx.kernel = !{!15}
!pacxx.kernel._ZN5pacxx2v213genericKernelILm0EZ4mainE12$_3675094202JPfS3_S3_iEEEvT0_DpNSt3__111conditionalIXsr3std12is_referenceIT1_EE5valueENS5_20add_lvalue_referenceINS0_17generic_to_globalIS7_E4typeEE4typeESB_E4typeE = !{!16}

!0 = !{void (i8, float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32)* @"_ZN5pacxx2v213genericKernelILm0EZ4mainE12$_3675094202JPfS3_S3_iEEEvT0_DpNSt3__111conditionalIXsr3std12is_referenceIT1_EE5valueENS5_20add_lvalue_referenceINS0_17generic_to_globalIS7_E4typeEE4typeESB_E4typeE", !"kernel", i32 1}
!1 = !{null, !"align", i32 8}
!2 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!3 = !{null, !"align", i32 16}
!4 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!5 = !{void (i8, float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32)* @"_ZN5pacxx2v213genericKernelILm0EZ4mainE12$_3675094202JPfS3_S3_iEEEvT0_DpNSt3__111conditionalIXsr3std12is_referenceIT1_EE5valueENS5_20add_lvalue_referenceINS0_17generic_to_globalIS7_E4typeEE4typeESB_E4typeE", !6, !7, !8, !9, !10, !11}
!6 = !{!"kernel_arg_addr_space", i32 0, i32 1, i32 1, i32 1, i32 0}
!7 = !{!"kernel_arg_type", !"class (lambda at Native.cpp:55:15)", !"std::conditional_t<std::is_reference<float*>::value, std::add_lvalue_reference_t<typename generic_to_global<float *>::type>, typename generic_to_global<float *>::type>", !"std::conditional_t<std::is_reference<float*>::value, std::add_lvalue_reference_t<typename generic_to_global<float *>::type>, typename generic_to_global<float *>::type>", !"std::conditional_t<std::is_reference<float*>::value, std::add_lvalue_reference_t<typename generic_to_global<float *>::type>, typename generic_to_global<float *>::type>", !"std::conditional_t<std::is_reference<int>::value, std::add_lvalue_reference_t<typename generic_to_global<int>::type>, typename generic_to_global<int>::type>"}
!8 = !{!"kernel_arg_name", !"callable", !"args", !"args", !"args", !"args"}
!9 = !{!"kernel_arg_access_qual", !"", !"", !"", !"", !""}
!10 = !{!"kernel_arg_base_type", !"class (lambda at Native.cpp:55:15)", !" float*", !" float*", !" float*", !"int"}
!11 = !{!"kernel_arg_type_qual", !"", !"", !"", !"", !""}
!12 = !{i32 1, i32 2}
!13 = !{}
!14 = !{!"clang version 3.8.0 (https://lklein14@bitbucket.org/mhaidl/clang.git 35a35447d041832b6e2e25acaf7c825860f8f407) (https://lklein14@bitbucket.org/lklein14/llvm.git 1038c73267508716cd4557f0233750e75c900f93)"}
!15 = !{void (i8, float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32)* @"_ZN5pacxx2v213genericKernelILm0EZ4mainE12$_3675094202JPfS3_S3_iEEEvT0_DpNSt3__111conditionalIXsr3std12is_referenceIT1_EE5valueENS5_20add_lvalue_referenceINS0_17generic_to_globalIS7_E4typeEE4typeESB_E4typeE"}
!16 = !{i32 -1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 -1, i32 0, i32 0}
!17 = !{i64 0}
