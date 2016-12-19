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

define ptx_kernel void @"_ZN5pacxx2v213genericKernelILm0EZ4mainE12$_4028513607JPfS3_S3_iEEEvT0_DpNSt3__111conditionalIXsr3std12is_referenceIT1_EE5valueENS5_20add_lvalue_referenceINS0_17generic_to_globalIS7_E4typeEE4typeESB_E4typeE"(i8 %callable.coerce, float addrspace(1)* %args, float addrspace(1)* %args1, float addrspace(1)* %args2, i32 %args3) {
  %1 = call spir_func i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1
  %2 = call spir_func i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1
  %3 = call spir_func i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1
  %4 = mul i32 %2, %1
  %5 = add i32 %4, %3
  %6 = icmp ult i32 %5, %args3
  br i1 %6, label %7, label %"_ZZ4mainENK12$_4028513607clEPKfS1_Pfj.exit"

; <label>:7                                       ; preds = %0
  %8 = sext i32 %5 to i64
  %9 = getelementptr float, float addrspace(1)* %args, i64 %8
  %10 = ptrtoint float addrspace(1)* %9 to i64, !pacxx.addrspace !18
  %11 = inttoptr i64 %10 to float*
  %12 = getelementptr float, float addrspace(1)* %args1, i64 %8
  %13 = ptrtoint float addrspace(1)* %12 to i64, !pacxx.addrspace !18
  %14 = inttoptr i64 %13 to float*
  %15 = load float, float* %11, align 4
  %16 = load float, float* %14, align 4
  %17 = fadd float %15, %16
  %18 = getelementptr float, float addrspace(1)* %args2, i64 %8
  %19 = ptrtoint float addrspace(1)* %18 to i64, !pacxx.addrspace !18
  %20 = inttoptr i64 %19 to float*
  store float %17, float* %20, align 4
  br label %"_ZZ4mainENK12$_4028513607clEPKfS1_Pfj.exit"

"_ZZ4mainENK12$_4028513607clEPKfS1_Pfj.exit":     ; preds = %7, %0
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
!pacxx.kernel._ZN5pacxx2v213genericKernelILm0EZ4mainE12$_4028513607JPfS3_S3_iEEEvT0_DpNSt3__111conditionalIXsr3std12is_referenceIT1_EE5valueENS5_20add_lvalue_referenceINS0_17generic_to_globalIS7_E4typeEE4typeESB_E4typeE = !{!17}
!nvvm.internalize.after.link = !{}
!nvvmir.version = !{!14}

!0 = !{!"clang version 3.8.0 (https://lklein14@bitbucket.org/mhaidl/clang.git 35a35447d041832b6e2e25acaf7c825860f8f407) (https://lklein14@bitbucket.org/lklein14/llvm.git 0645d2ecef7e124b06d0d0e4d14ec719b5e437f6)"}
!1 = !{!"clang version 3.8.0 (https://lklein14@bitbucket.org/mhaidl/clang.git 35a35447d041832b6e2e25acaf7c825860f8f407) (https://lklein14@bitbucket.org/lklein14/llvm.git 95e55b0d66d7a0c9851bbf090a9e63c01930a26a)"}
!2 = !{void (i8, float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32)* @"_ZN5pacxx2v213genericKernelILm0EZ4mainE12$_4028513607JPfS3_S3_iEEEvT0_DpNSt3__111conditionalIXsr3std12is_referenceIT1_EE5valueENS5_20add_lvalue_referenceINS0_17generic_to_globalIS7_E4typeEE4typeESB_E4typeE", !"kernel", i32 1}
!3 = !{null, !"align", i32 8}
!4 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!5 = !{null, !"align", i32 16}
!6 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!7 = !{void (i8, float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32)* @"_ZN5pacxx2v213genericKernelILm0EZ4mainE12$_4028513607JPfS3_S3_iEEEvT0_DpNSt3__111conditionalIXsr3std12is_referenceIT1_EE5valueENS5_20add_lvalue_referenceINS0_17generic_to_globalIS7_E4typeEE4typeESB_E4typeE", !8, !9, !10, !11, !12, !13}
!8 = !{!"kernel_arg_addr_space", i32 0, i32 1, i32 1, i32 1, i32 0}
!9 = !{!"kernel_arg_type", !"class (lambda at vectorAdd.cpp:60:15)", !"std::conditional_t<std::is_reference<float*>::value, std::add_lvalue_reference_t<typename generic_to_global<float *>::type>, typename generic_to_global<float *>::type>", !"std::conditional_t<std::is_reference<float*>::value, std::add_lvalue_reference_t<typename generic_to_global<float *>::type>, typename generic_to_global<float *>::type>", !"std::conditional_t<std::is_reference<float*>::value, std::add_lvalue_reference_t<typename generic_to_global<float *>::type>, typename generic_to_global<float *>::type>", !"std::conditional_t<std::is_reference<int>::value, std::add_lvalue_reference_t<typename generic_to_global<int>::type>, typename generic_to_global<int>::type>"}
!10 = !{!"kernel_arg_name", !"callable", !"args", !"args", !"args", !"args"}
!11 = !{!"kernel_arg_access_qual", !"", !"", !"", !"", !""}
!12 = !{!"kernel_arg_base_type", !"class (lambda at vectorAdd.cpp:60:15)", !" float*", !" float*", !" float*", !"int"}
!13 = !{!"kernel_arg_type_qual", !"", !"", !"", !"", !""}
!14 = !{i32 1, i32 2}
!15 = !{}
!16 = !{void (i8, float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i32)* @"_ZN5pacxx2v213genericKernelILm0EZ4mainE12$_4028513607JPfS3_S3_iEEEvT0_DpNSt3__111conditionalIXsr3std12is_referenceIT1_EE5valueENS5_20add_lvalue_referenceINS0_17generic_to_globalIS7_E4typeEE4typeESB_E4typeE"}
!17 = !{i32 -1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 -1, i32 0, i32 0}
!18 = !{i64 0}
