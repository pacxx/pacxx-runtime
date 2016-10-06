; ModuleID = 'kernel.ll'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-unknown-unknown"

%"class.std::__1::vector" = type { %"class.std::__1::__vector_base" }
%"class.std::__1::__vector_base" = type { i32*, i32*, i8, i8, %"class.std::__1::__compressed_pair" }
%"class.std::__1::__compressed_pair" = type { %"class.std::__1::__libcpp_compressed_pair_imp" }
%"class.std::__1::__libcpp_compressed_pair_imp" = type { i32* }

@__cudart_i2opi_f = internal addrspace(4) global [6 x i32] [i32 1011060801, i32 -614296167, i32 -181084736, i32 -64530479, i32 1313084713, i32 -1560706194], align 4
@__cudart_i2opi_d = internal addrspace(4) global [18 x i64] [i64 7780917995555872008, i64 4397547296490951402, i64 8441921394348257659, i64 5712322887342352941, i64 7869616827067468215, i64 -1211730484530615009, i64 2303758334597371919, i64 -7168499653074671557, i64 4148332274289687028, i64 -1613291254968254911, i64 -1692731182770600828, i64 -135693905287338178, i64 452944820249399836, i64 -5249950069107600672, i64 -121206125134887583, i64 -2638381946312093631, i64 -277156292786332224, i64 -6703182060581546711], align 8

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0

; Function Attrs: nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

define ptx_kernel void @"_ZN5pacxx2v213genericKernelILm0EZ4mainE12$_2849746516JRNSt3__16vectorIiNS3_9allocatorIiEEEES8_S8_EEEvT0_DpNS3_11conditionalIXsr3std12is_referenceIT1_EE5valueENS3_20add_lvalue_referenceINS0_17generic_to_globalISB_E4typeEE4typeESF_E4typeE"(i8 %callable.coerce, %"class.std::__1::vector" addrspace(1)* %args, %"class.std::__1::vector" addrspace(1)* %args1, %"class.std::__1::vector" addrspace(1)* %args2) {
  %1 = call spir_func i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1
  %2 = call spir_func i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1
  %3 = call spir_func i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1
  %4 = mul i32 %2, %1
  %5 = add i32 %4, %3
  %6 = sext i32 %5 to i64
  %7 = getelementptr %"class.std::__1::vector", %"class.std::__1::vector" addrspace(1)* %args, i64 -1, i32 0, i32 1
  %8 = bitcast i32* addrspace(1)* %7 to i64 addrspace(1)*
  %9 = ptrtoint i64 addrspace(1)* %8 to i64, !pacxx.addrspace !17
  %10 = inttoptr i64 %9 to i64*
  %11 = getelementptr %"class.std::__1::vector", %"class.std::__1::vector" addrspace(1)* %args, i64 -1
  %12 = bitcast %"class.std::__1::vector" addrspace(1)* %11 to i64 addrspace(1)*
  %13 = ptrtoint i64 addrspace(1)* %12 to i64, !pacxx.addrspace !17
  %14 = inttoptr i64 %13 to i64*
  %15 = load i64, i64* %10, align 8
  %16 = load i64, i64* %14, align 8
  %17 = sub i64 %15, %16
  %18 = lshr i64 %17, 2
  %19 = icmp ult i64 %6, %18
  br i1 %19, label %20, label %"_ZZ4mainENK12$_2849746516clERKNSt3__16vectorIiNS0_9allocatorIiEEEES6_RS4_.exit"

; <label>:20                                      ; preds = %0
  %21 = bitcast %"class.std::__1::vector" addrspace(1)* %args to i32 addrspace(1)*
  %22 = ptrtoint i32 addrspace(1)* %21 to i64, !pacxx.addrspace !17
  %23 = inttoptr i64 %22 to i32*
  %24 = getelementptr i32, i32 addrspace(1)* %21, i64 %6
  %25 = bitcast %"class.std::__1::vector" addrspace(1)* %args1 to i32 addrspace(1)*
  %26 = ptrtoint i32 addrspace(1)* %25 to i64, !pacxx.addrspace !17
  %27 = inttoptr i64 %26 to i32*
  %28 = getelementptr i32, i32 addrspace(1)* %25, i64 %6
  %29 = load i32, i32 addrspace(1)* %24
  %30 = load i32, i32 addrspace(1)* %28
  %31 = add nsw i32 %30, %29
  %32 = bitcast %"class.std::__1::vector" addrspace(1)* %args2 to i32 addrspace(1)*
  %33 = ptrtoint i32 addrspace(1)* %32 to i64, !pacxx.addrspace !17
  %34 = inttoptr i64 %33 to i32*
  %35 = getelementptr i32, i32 addrspace(1)* %32, i64 %6
  store i32 %31, i32 addrspace(1)* %35
  br label %"_ZZ4mainENK12$_2849746516clERKNSt3__16vectorIiNS0_9allocatorIiEEEES6_RS4_.exit"

"_ZZ4mainENK12$_2849746516clERKNSt3__16vectorIiNS0_9allocatorIiEEEES6_RS4_.exit": ; preds = %20, %0
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }

!llvm.ident = !{!0, !0, !0}
!nvvm.annotations = !{!1, !2, !3, !2, !4, !4, !4, !4, !5, !5, !4}
!opencl.kernels = !{!6}
!opencl.spir.version = !{!13}
!opencl.ocl.version = !{!13}
!opencl.enable.FP_CONTRACT = !{!14}
!opencl.used.optional.core.features = !{!14}
!opencl.used.extensions = !{!14}
!opencl.compiler.options = !{!14}
!pacxx.kernel = !{!15}
!pacxx.kernel._ZN5pacxx2v213genericKernelILm0EZ4mainE12$_2849746516JRNSt3__16vectorIiNS3_9allocatorIiEEEES8_S8_EEEvT0_DpNS3_11conditionalIXsr3std12is_referenceIT1_EE5valueENS3_20add_lvalue_referenceINS0_17generic_to_globalISB_E4typeEE4typeESF_E4typeE = !{!16}
!nvvm.internalize.after.link = !{}
!nvvmir.version = !{!13}

!0 = !{!"clang version 3.8.0 (https://lklein14@bitbucket.org/mhaidl/clang.git 35a35447d041832b6e2e25acaf7c825860f8f407) (https://lklein14@bitbucket.org/mhaidl/llvm.git e40887d4d85032687644dd45f7e8835c7e327fe2)"}
!1 = !{void (i8, %"class.std::__1::vector" addrspace(1)*, %"class.std::__1::vector" addrspace(1)*, %"class.std::__1::vector" addrspace(1)*)* @"_ZN5pacxx2v213genericKernelILm0EZ4mainE12$_2849746516JRNSt3__16vectorIiNS3_9allocatorIiEEEES8_S8_EEEvT0_DpNS3_11conditionalIXsr3std12is_referenceIT1_EE5valueENS3_20add_lvalue_referenceINS0_17generic_to_globalISB_E4typeEE4typeESF_E4typeE", !"kernel", i32 1}
!2 = !{null, !"align", i32 8}
!3 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!4 = !{null, !"align", i32 16}
!5 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!6 = !{void (i8, %"class.std::__1::vector" addrspace(1)*, %"class.std::__1::vector" addrspace(1)*, %"class.std::__1::vector" addrspace(1)*)* @"_ZN5pacxx2v213genericKernelILm0EZ4mainE12$_2849746516JRNSt3__16vectorIiNS3_9allocatorIiEEEES8_S8_EEEvT0_DpNS3_11conditionalIXsr3std12is_referenceIT1_EE5valueENS3_20add_lvalue_referenceINS0_17generic_to_globalISB_E4typeEE4typeESF_E4typeE", !7, !8, !9, !10, !11, !12}
!7 = !{!"kernel_arg_addr_space", i32 0, i32 1, i32 1, i32 1}
!8 = !{!"kernel_arg_type", !"class (lambda at Vector.cpp:15:20)", !"std::conditional_t<std::is_reference<vector<int, allocator<int> > &>::value, std::add_lvalue_reference_t<typename generic_to_global<vector<int, allocator<int> > &>::type>, typename generic_to_global<vector<int, allocator<int> > &>::type>", !"std::conditional_t<std::is_reference<vector<int, allocator<int> > &>::value, std::add_lvalue_reference_t<typename generic_to_global<vector<int, allocator<int> > &>::type>, typename generic_to_global<vector<int, allocator<int> > &>::type>", !"std::conditional_t<std::is_reference<vector<int, allocator<int> > &>::value, std::add_lvalue_reference_t<typename generic_to_global<vector<int, allocator<int> > &>::type>, typename generic_to_global<vector<int, allocator<int> > &>::type>"}
!9 = !{!"kernel_arg_name", !"callable", !"args", !"args", !"args"}
!10 = !{!"kernel_arg_access_qual", !"", !"", !"", !""}
!11 = !{!"kernel_arg_base_type", !"class (lambda at Vector.cpp:15:20)", !"void*", !"void*", !"void*"}
!12 = !{!"kernel_arg_type_qual", !"", !"", !"", !""}
!13 = !{i32 1, i32 2}
!14 = !{}
!15 = !{void (i8, %"class.std::__1::vector" addrspace(1)*, %"class.std::__1::vector" addrspace(1)*, %"class.std::__1::vector" addrspace(1)*)* @"_ZN5pacxx2v213genericKernelILm0EZ4mainE12$_2849746516JRNSt3__16vectorIiNS3_9allocatorIiEEEES8_S8_EEEvT0_DpNS3_11conditionalIXsr3std12is_referenceIT1_EE5valueENS3_20add_lvalue_referenceINS0_17generic_to_globalISB_E4typeEE4typeESF_E4typeE"}
!16 = !{i32 -1, i32 0, i32 0, i32 -1, i32 0, i32 0, i32 -1, i32 0, i32 0, i32 -1, i32 0, i32 0}
!17 = !{i64 0}
