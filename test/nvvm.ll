; ModuleID = './tmp-161005-1101-fJCEta/nvvm.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-unknown-unknown"

%"class.std::__1::vector" = type { %"class.std::__1::__vector_base" }
%"class.std::__1::__vector_base" = type { i32*, i32*, i8, i8, %"class.std::__1::__compressed_pair" }
%"class.std::__1::__compressed_pair" = type { %"class.std::__1::__libcpp_compressed_pair_imp" }
%"class.std::__1::__libcpp_compressed_pair_imp" = type { i32* }

declare i32 @vprintf(i8* nocapture, i8*)

define void @__printf(i8 addrspace(4)* %ptr, i8* %val) {
entry:
  %0 = addrspacecast i8 addrspace(4)* %ptr to i8*
  %call = call i32 @vprintf(i8* %0, i8* %val)
  ret void
}

define ptx_kernel void @"_ZN5pacxx2v213genericKernelILm0EZ4mainE12$_2849746516JRNSt3__16vectorIiNS3_9allocatorIiEEEES8_S8_EEEvT0_DpNS3_11conditionalIXsr3std12is_referenceIT1_EE5valueENS3_20add_lvalue_referenceINS0_17generic_to_globalISB_E4typeEE4typeESF_E4typeE"(i8 %callable.coerce, %"class.std::__1::vector" addrspace(1)* %args, %"class.std::__1::vector" addrspace(1)* %args1, %"class.std::__1::vector" addrspace(1)* %args2) {
_Z13get_global_idj.exit:
  %0 = ptrtoint %"class.std::__1::vector" addrspace(1)* %args to i64, !pacxx.addrspace !17
  %1 = inttoptr i64 %0 to %"class.std::__1::vector"*
  %2 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1
  %3 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1
  %4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1
  %5 = mul i32 %2, %3
  %6 = add i32 %5, %4
  %7 = sext i32 %6 to i64
  %8 = getelementptr inbounds %"class.std::__1::vector", %"class.std::__1::vector"* %1, i64 -1, i32 0, i32 1
  %9 = getelementptr %"class.std::__1::vector", %"class.std::__1::vector" addrspace(1)* %args, i64 -1
  %10 = ptrtoint %"class.std::__1::vector" addrspace(1)* %9 to i64
  %11 = inttoptr i64 %10 to i64*
  %12 = bitcast i32** %8 to i64*
  %13 = load i64, i64* %12, align 8
  %14 = load i64, i64* %11, align 8
  %15 = sub i64 %13, %14
  %16 = lshr i64 %15, 2
  %17 = icmp ult i64 %7, %16
  br i1 %17, label %18, label %"_ZZ4mainENK12$_2849746516clERKNSt3__16vectorIiNS0_9allocatorIiEEEES6_RS4_.exit"

; <label>:18                                      ; preds = %_Z13get_global_idj.exit
  %19 = bitcast %"class.std::__1::vector" addrspace(1)* %args to i32 addrspace(1)*
  %20 = getelementptr i32, i32 addrspace(1)* %19, i64 %7
  %21 = bitcast %"class.std::__1::vector" addrspace(1)* %args1 to i32 addrspace(1)*
  %22 = getelementptr i32, i32 addrspace(1)* %21, i64 %7
  %23 = load i32, i32 addrspace(1)* %20, align 4
  %24 = load i32, i32 addrspace(1)* %22, align 4
  %25 = add nsw i32 %24, %23
  %26 = bitcast %"class.std::__1::vector" addrspace(1)* %args2 to i32 addrspace(1)*
  %27 = getelementptr i32, i32 addrspace(1)* %26, i64 %7
  store i32 %25, i32 addrspace(1)* %27, align 4
  br label %"_ZZ4mainENK12$_2849746516clERKNSt3__16vectorIiNS0_9allocatorIiEEEES6_RS4_.exit"

"_ZZ4mainENK12$_2849746516clERKNSt3__16vectorIiNS0_9allocatorIiEEEES6_RS4_.exit": ; preds = %_Z13get_global_idj.exit, %18
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
!pacxx.kernel._ZN5pacxx2v213genericKernelILm0EZ4mainE12$_2849746516JRNSt3__16vectorIiNS3_9allocatorIiEEEES8_S8_EEEvT0_DpNS3_11conditionalIXsr3std12is_referenceIT1_EE5valueENS3_20add_lvalue_referenceINS0_17generic_to_globalISB_E4typeEE4typeESF_E4typeE = !{!16}

!0 = !{void (i8, %"class.std::__1::vector" addrspace(1)*, %"class.std::__1::vector" addrspace(1)*, %"class.std::__1::vector" addrspace(1)*)* @"_ZN5pacxx2v213genericKernelILm0EZ4mainE12$_2849746516JRNSt3__16vectorIiNS3_9allocatorIiEEEES8_S8_EEEvT0_DpNS3_11conditionalIXsr3std12is_referenceIT1_EE5valueENS3_20add_lvalue_referenceINS0_17generic_to_globalISB_E4typeEE4typeESF_E4typeE", !"kernel", i32 1}
!1 = !{null, !"align", i32 8}
!2 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!3 = !{null, !"align", i32 16}
!4 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
!5 = !{void (i8, %"class.std::__1::vector" addrspace(1)*, %"class.std::__1::vector" addrspace(1)*, %"class.std::__1::vector" addrspace(1)*)* @"_ZN5pacxx2v213genericKernelILm0EZ4mainE12$_2849746516JRNSt3__16vectorIiNS3_9allocatorIiEEEES8_S8_EEEvT0_DpNS3_11conditionalIXsr3std12is_referenceIT1_EE5valueENS3_20add_lvalue_referenceINS0_17generic_to_globalISB_E4typeEE4typeESF_E4typeE", !6, !7, !8, !9, !10, !11}
!6 = !{!"kernel_arg_addr_space", i32 0, i32 1, i32 1, i32 1}
!7 = !{!"kernel_arg_type", !"class (lambda at Vector.cpp:15:20)", !"std::conditional_t<std::is_reference<vector<int, allocator<int> > &>::value, std::add_lvalue_reference_t<typename generic_to_global<vector<int, allocator<int> > &>::type>, typename generic_to_global<vector<int, allocator<int> > &>::type>", !"std::conditional_t<std::is_reference<vector<int, allocator<int> > &>::value, std::add_lvalue_reference_t<typename generic_to_global<vector<int, allocator<int> > &>::type>, typename generic_to_global<vector<int, allocator<int> > &>::type>", !"std::conditional_t<std::is_reference<vector<int, allocator<int> > &>::value, std::add_lvalue_reference_t<typename generic_to_global<vector<int, allocator<int> > &>::type>, typename generic_to_global<vector<int, allocator<int> > &>::type>"}
!8 = !{!"kernel_arg_name", !"callable", !"args", !"args", !"args"}
!9 = !{!"kernel_arg_access_qual", !"", !"", !"", !""}
!10 = !{!"kernel_arg_base_type", !"class (lambda at Vector.cpp:15:20)", !"void*", !"void*", !"void*"}
!11 = !{!"kernel_arg_type_qual", !"", !"", !"", !""}
!12 = !{i32 1, i32 2}
!13 = !{}
!14 = !{!"clang version 3.8.0 (https://lklein14@bitbucket.org/mhaidl/clang.git 35a35447d041832b6e2e25acaf7c825860f8f407) (https://lklein14@bitbucket.org/mhaidl/llvm.git e40887d4d85032687644dd45f7e8835c7e327fe2)"}
!15 = !{void (i8, %"class.std::__1::vector" addrspace(1)*, %"class.std::__1::vector" addrspace(1)*, %"class.std::__1::vector" addrspace(1)*)* @"_ZN5pacxx2v213genericKernelILm0EZ4mainE12$_2849746516JRNSt3__16vectorIiNS3_9allocatorIiEEEES8_S8_EEEvT0_DpNS3_11conditionalIXsr3std12is_referenceIT1_EE5valueENS3_20add_lvalue_referenceINS0_17generic_to_globalISB_E4typeEE4typeESF_E4typeE"}
!16 = !{i32 -1, i32 0, i32 0, i32 -1, i32 0, i32 0, i32 -1, i32 0, i32 0, i32 -1, i32 0, i32 0}
!17 = !{i64 0}
