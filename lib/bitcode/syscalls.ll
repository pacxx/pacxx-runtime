target triple = "nvptx64-unknown-unknown"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

declare i32 @vprintf(i8* nocapture, i8*)

define void @__printf(i8 addrspace(4)* %ptr, i8* %val){
entry:
  %0 = addrspacecast i8 addrspace(4)* %ptr to i8*
  %call = call i32 @vprintf(i8* %0, i8* %val)
  ret void
}

define void @_printf(i8* %ptr, i8* %val){
entry:
  %0 = addrspacecast i8* %ptr to i8 addrspace(4)*
  call void @__printf(i8 addrspace(4)* %0, i8* %val)
  ret void
}