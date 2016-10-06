; ModuleID = './tmp-161005-1101-fJCEta/pacxx.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "pacxx-unknown-unknown"

%"class.std::__1::vector" = type { %"class.std::__1::__vector_base" }
%"class.std::__1::__vector_base" = type { i32*, i32*, i8, i8, %"class.std::__1::__compressed_pair" }
%"class.std::__1::__compressed_pair" = type { %"class.std::__1::__libcpp_compressed_pair_imp" }
%"class.std::__1::__libcpp_compressed_pair_imp" = type { i32* }
%struct.Thread = type { %struct._idx, %struct._idx }
%struct._idx = type { i32, i32, i32 }

; Function Attrs: noinline
define spir_kernel void @"_ZN5pacxx2v213genericKernelILm0EZ4mainE12$_2849746516JRNSt3__16vectorIiNS3_9allocatorIiEEEES8_S8_EEEvT0_DpNS3_11conditionalIXsr3std12is_referenceIT1_EE5valueENS3_20add_lvalue_referenceINS0_17generic_to_globalISB_E4typeEE4typeESF_E4typeE"(i8 %callable.coerce, %"class.std::__1::vector" addrspace(1)* nocapture readonly dereferenceable(32) %args, %"class.std::__1::vector" addrspace(1)* nocapture readonly dereferenceable(32) %args1, %"class.std::__1::vector" addrspace(1)* nocapture dereferenceable(32) %args2) #0 {
  %1 = addrspacecast %"class.std::__1::vector" addrspace(1)* %args to %"class.std::__1::vector"*
  %2 = tail call i32 @_Z12get_local_idj(i32 0)
  %3 = tail call i32 @_Z12get_local_idj(i32 1)
  %4 = tail call i32 @_Z12get_local_idj(i32 2)
  %5 = tail call i32 @_Z13get_global_idj(i32 0)
  %6 = tail call i32 @_Z13get_global_idj(i32 1)
  %7 = tail call i32 @_Z13get_global_idj(i32 2)
  %8 = sext i32 %5 to i64
  %9 = getelementptr inbounds %"class.std::__1::vector", %"class.std::__1::vector"* %1, i64 -1, i32 0, i32 1
  %10 = bitcast i32** %9 to i64*
  %11 = load i64, i64* %10, align 8
  %12 = getelementptr inbounds %"class.std::__1::vector", %"class.std::__1::vector" addrspace(1)* %args, i64 -1
  %13 = bitcast %"class.std::__1::vector" addrspace(1)* %12 to i64 addrspace(1)*
  %14 = addrspacecast i64 addrspace(1)* %13 to i64*
  %15 = load i64, i64* %14, align 8
  %16 = sub i64 %11, %15
  %17 = lshr i64 %16, 2
  %18 = icmp ult i64 %8, %17
  br i1 %18, label %19, label %"_ZZ4mainENK12$_2849746516clERKNSt3__16vectorIiNS0_9allocatorIiEEEES6_RS4_.exit"

; <label>:19                                      ; preds = %0
  %20 = bitcast %"class.std::__1::vector" addrspace(1)* %args to i32 addrspace(1)*
  %21 = addrspacecast i32 addrspace(1)* %20 to i32*
  %22 = getelementptr inbounds i32, i32* %21, i64 %8
  %23 = load i32, i32* %22, align 4
  %24 = bitcast %"class.std::__1::vector" addrspace(1)* %args1 to i32 addrspace(1)*
  %25 = addrspacecast i32 addrspace(1)* %24 to i32*
  %26 = getelementptr inbounds i32, i32* %25, i64 %8
  %27 = load i32, i32* %26, align 4
  %28 = add nsw i32 %27, %23
  %29 = bitcast %"class.std::__1::vector" addrspace(1)* %args2 to i32 addrspace(1)*
  %30 = addrspacecast i32 addrspace(1)* %29 to i32*
  %31 = getelementptr inbounds i32, i32* %30, i64 %8
  store i32 %28, i32* %31, align 4
  br label %"_ZZ4mainENK12$_2849746516clERKNSt3__16vectorIiNS0_9allocatorIiEEEES6_RS4_.exit"

"_ZZ4mainENK12$_2849746516clERKNSt3__16vectorIiNS0_9allocatorIiEEEES6_RS4_.exit": ; preds = %0, %19
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
