; ModuleID = './tmp-161005-1101-fJCEta/spir.bc'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%"class.std::__1::vector" = type { %"class.std::__1::__vector_base" }
%"class.std::__1::__vector_base" = type { i32*, i32*, i8, i8, %"class.std::__1::__compressed_pair" }
%"class.std::__1::__compressed_pair" = type { %"class.std::__1::__libcpp_compressed_pair_imp" }
%"class.std::__1::__libcpp_compressed_pair_imp" = type { i32* }

!nvvm.annotations = !{!0}
!opencl.kernels = !{!1}
!opencl.spir.version = !{!8}
!opencl.ocl.version = !{!8}
!opencl.enable.FP_CONTRACT = !{!9}
!opencl.used.optional.core.features = !{!9}
!opencl.used.extensions = !{!9}
!opencl.compiler.options = !{!9}
!llvm.ident = !{!10, !10}

!0 = !{void (i8, %"class.std::__1::vector" addrspace(1)*, %"class.std::__1::vector" addrspace(1)*, %"class.std::__1::vector" addrspace(1)*)* undef, !"kernel", i32 1}
!1 = !{void (i8, %"class.std::__1::vector" addrspace(1)*, %"class.std::__1::vector" addrspace(1)*, %"class.std::__1::vector" addrspace(1)*)* undef, !2, !3, !4, !5, !6, !7}
!2 = !{!"kernel_arg_addr_space", i32 0, i32 1, i32 1, i32 1}
!3 = !{!"kernel_arg_type", !"class (lambda at Vector.cpp:15:20)", !"std::conditional_t<std::is_reference<vector<int, allocator<int> > &>::value, std::add_lvalue_reference_t<typename generic_to_global<vector<int, allocator<int> > &>::type>, typename generic_to_global<vector<int, allocator<int> > &>::type>", !"std::conditional_t<std::is_reference<vector<int, allocator<int> > &>::value, std::add_lvalue_reference_t<typename generic_to_global<vector<int, allocator<int> > &>::type>, typename generic_to_global<vector<int, allocator<int> > &>::type>", !"std::conditional_t<std::is_reference<vector<int, allocator<int> > &>::value, std::add_lvalue_reference_t<typename generic_to_global<vector<int, allocator<int> > &>::type>, typename generic_to_global<vector<int, allocator<int> > &>::type>"}
!4 = !{!"kernel_arg_name", !"callable", !"args", !"args", !"args"}
!5 = !{!"kernel_arg_access_qual", !"", !"", !"", !""}
!6 = !{!"kernel_arg_base_type", !"class (lambda at Vector.cpp:15:20)", !"void*", !"void*", !"void*"}
!7 = !{!"kernel_arg_type_qual", !"", !"", !"", !""}
!8 = !{i32 1, i32 2}
!9 = !{}
!10 = !{!"clang version 3.8.0 (https://lklein14@bitbucket.org/mhaidl/clang.git 35a35447d041832b6e2e25acaf7c825860f8f407) (https://lklein14@bitbucket.org/mhaidl/llvm.git e40887d4d85032687644dd45f7e8835c7e327fe2)"}
