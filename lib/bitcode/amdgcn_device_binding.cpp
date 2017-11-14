//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

unsigned __ockl_get_num_groups(unsigned dim);
unsigned __ockl_get_local_size(unsigned dim);

unsigned __pacxx_get_num_groups(unsigned dim){ 
    return __ockl_get_num_groups(dim);
}

unsigned __pacxx_get_local_size(unsigned dim){
    return __ockl_get_local_size(dim);
}