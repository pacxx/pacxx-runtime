//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

extern "C" {
unsigned __ockl_get_num_groups(unsigned dim);
unsigned __ockl_get_local_size(unsigned dim);
float __ocml_exp_f32(float value);
float __ocml_sqrt_f32(float value);

__attribute__((address_space(3))) void* get_dynamic_group_segment_base_pointer();

unsigned __pacxx_get_num_groups(unsigned dim){ 
    return __ockl_get_num_groups(dim);
}

unsigned __pacxx_get_local_size(unsigned dim){
    return __ockl_get_local_size(dim);
}

__attribute__((address_space(3))) void* __get_extern_shared_mem_ptr(){
  return get_dynamic_group_segment_base_pointer();
}

float sqrtf(float value){
    return __ocml_sqrt_f32(value);
}

float expf(float value){
    return __ocml_exp_f32(value);
}

}
