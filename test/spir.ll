; ModuleID = './tmp-161005-1101-fJCEta/spir.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir64-unknown-unknown"

%"class.std::__1::vector" = type { %"class.std::__1::__vector_base" }
%"class.std::__1::__vector_base" = type { i32*, i32*, i8, i8, %"class.std::__1::__compressed_pair" }
%"class.std::__1::__compressed_pair" = type { %"class.std::__1::__libcpp_compressed_pair_imp" }
%"class.std::__1::__libcpp_compressed_pair_imp" = type { i32* }

define spir_kernel void @"_ZN5pacxx2v213genericKernelILm0EZ4mainE12$_2849746516JRNSt3__16vectorIiNS3_9allocatorIiEEEES8_S8_EEEvT0_DpNS3_11conditionalIXsr3std12is_referenceIT1_EE5valueENS3_20add_lvalue_referenceINS0_17generic_to_globalISB_E4typeEE4typeESF_E4typeE"(i8 %callable.coerce, %"class.std::__1::vector" addrspace(1)* %args, %"class.std::__1::vector" addrspace(1)* %args1, %"class.std::__1::vector" addrspace(1)* %args2) {
  %1 = ptrtoint %"class.std::__1::vector" addrspace(1)* %args to i64, !pacxx.addrspace !11
  %2 = inttoptr i64 %1 to %"class.std::__1::vector"*
  %3 = addrspacecast %"class.std::__1::vector" addrspace(1)* %args to %"class.std::__1::vector"*
  %4 = tail call spir_func i32 @_Z13get_global_idj(i32 0)
  %5 = sext i32 %4 to i64
  %6 = getelementptr inbounds %"class.std::__1::vector", %"class.std::__1::vector"* %2, i64 -1, i32 0, i32 1
  %7 = getelementptr %"class.std::__1::vector", %"class.std::__1::vector" addrspace(1)* %args, i64 -1
  %8 = bitcast %"class.std::__1::vector" addrspace(1)* %7 to i64 addrspace(1)*
  %9 = ptrtoint i64 addrspace(1)* %8 to i64, !pacxx.addrspace !11
  %10 = inttoptr i64 %9 to i64*
  %11 = addrspacecast i64 addrspace(1)* %8 to i64*
  %12 = bitcast i32** %6 to i64*
  %13 = load i64, i64* %12, align 8
  %14 = load i64, i64* %10, align 8
  %15 = sub i64 %13, %14
  %16 = lshr i64 %15, 2
  %17 = icmp ult i64 %5, %16
  br i1 %17, label %18, label %"_ZZ4mainENK12$_2849746516clERKNSt3__16vectorIiNS0_9allocatorIiEEEES6_RS4_.exit"

; <label>:18                                      ; preds = %0
  %19 = bitcast %"class.std::__1::vector" addrspace(1)* %args to i32 addrspace(1)*
  %20 = ptrtoint i32 addrspace(1)* %19 to i64, !pacxx.addrspace !11
  %21 = inttoptr i64 %20 to i32*
  %22 = addrspacecast i32 addrspace(1)* %19 to i32*
  %23 = getelementptr inbounds i32, i32* %21, i64 %5
  %24 = getelementptr i32, i32 addrspace(1)* %19, i64 %5
  %25 = load i32, i32 addrspace(1)* %24, align 4
  %26 = bitcast %"class.std::__1::vector" addrspace(1)* %args1 to i32 addrspace(1)*
  %27 = ptrtoint i32 addrspace(1)* %26 to i64, !pacxx.addrspace !11
  %28 = inttoptr i64 %27 to i32*
  %29 = addrspacecast i32 addrspace(1)* %26 to i32*
  %30 = getelementptr inbounds i32, i32* %28, i64 %5
  %31 = getelementptr i32, i32 addrspace(1)* %26, i64 %5
  %32 = load i32, i32 addrspace(1)* %31, align 4
  %33 = load i32, i32 addrspace(1)* %24
  %34 = load i32, i32 addrspace(1)* %31
  %35 = add nsw i32 %34, %33
  %36 = bitcast %"class.std::__1::vector" addrspace(1)* %args2 to i32 addrspace(1)*
  %37 = ptrtoint i32 addrspace(1)* %36 to i64, !pacxx.addrspace !11
  %38 = inttoptr i64 %37 to i32*
  %39 = addrspacecast i32 addrspace(1)* %36 to i32*
  %40 = getelementptr inbounds i32, i32* %38, i64 %5
  %41 = getelementptr i32, i32 addrspace(1)* %36, i64 %5
  store i32 %35, i32 addrspace(1)* %41
  br label %"_ZZ4mainENK12$_2849746516clERKNSt3__16vectorIiNS0_9allocatorIiEEEES6_RS4_.exit"

"_ZZ4mainENK12$_2849746516clERKNSt3__16vectorIiNS0_9allocatorIiEEEES6_RS4_.exit": ; preds = %0, %18
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

!0 = !{void (i8, %"class.std::__1::vector" addrspace(1)*, %"class.std::__1::vector" addrspace(1)*, %"class.std::__1::vector" addrspace(1)*)* @"_ZN5pacxx2v213genericKernelILm0EZ4mainE12$_2849746516JRNSt3__16vectorIiNS3_9allocatorIiEEEES8_S8_EEEvT0_DpNS3_11conditionalIXsr3std12is_referenceIT1_EE5valueENS3_20add_lvalue_referenceINS0_17generic_to_globalISB_E4typeEE4typeESF_E4typeE", !"kernel", i32 1}
!1 = !{void (i8, %"class.std::__1::vector" addrspace(1)*, %"class.std::__1::vector" addrspace(1)*, %"class.std::__1::vector" addrspace(1)*)* @"_ZN5pacxx2v213genericKernelILm0EZ4mainE12$_2849746516JRNSt3__16vectorIiNS3_9allocatorIiEEEES8_S8_EEEvT0_DpNS3_11conditionalIXsr3std12is_referenceIT1_EE5valueENS3_20add_lvalue_referenceINS0_17generic_to_globalISB_E4typeEE4typeESF_E4typeE", !2, !3, !4, !5, !6, !7}
!2 = !{!"kernel_arg_addr_space", i32 0, i32 1, i32 1, i32 1}
!3 = !{!"kernel_arg_type", !"class (lambda at Vector.cpp:15:20)", !"std::conditional_t<std::is_reference<vector<int, allocator<int> > &>::value, std::add_lvalue_reference_t<typename generic_to_global<vector<int, allocator<int> > &>::type>, typename generic_to_global<vector<int, allocator<int> > &>::type>", !"std::conditional_t<std::is_reference<vector<int, allocator<int> > &>::value, std::add_lvalue_reference_t<typename generic_to_global<vector<int, allocator<int> > &>::type>, typename generic_to_global<vector<int, allocator<int> > &>::type>", !"std::conditional_t<std::is_reference<vector<int, allocator<int> > &>::value, std::add_lvalue_reference_t<typename generic_to_global<vector<int, allocator<int> > &>::type>, typename generic_to_global<vector<int, allocator<int> > &>::type>"}
!4 = !{!"kernel_arg_name", !"callable", !"args", !"args", !"args"}
!5 = !{!"kernel_arg_access_qual", !"", !"", !"", !""}
!6 = !{!"kernel_arg_base_type", !"class (lambda at Vector.cpp:15:20)", !"void*", !"void*", !"void*"}
!7 = !{!"kernel_arg_type_qual", !"", !"", !"", !""}
!8 = !{i32 1, i32 2}
!9 = !{}
!10 = !{!"clang version 3.8.0 (https://lklein14@bitbucket.org/mhaidl/clang.git 35a35447d041832b6e2e25acaf7c825860f8f407) (https://lklein14@bitbucket.org/mhaidl/llvm.git e40887d4d85032687644dd45f7e8835c7e327fe2)"}
!11 = !{i64 0}
