//
// Created by mhaidl on 30/05/16.
//
#include "pacxx/detail/rocm/HIPErrorDetection.h"
#include "pacxx/detail/common/Common.h"
#include "pacxx/detail/common/Exceptions.h"

#define __HIP_PLATFORM_HCC__
#include <hip/hip_runtime.h>


namespace pacxx {

namespace v2 {

bool checkHIPCall(hipError_t result, char const *const func,
                   const char *const file, int const line) {
  if (result != hipError_t::hipSuccess) {
    throw common::generic_exception(
        common::to_string("HIP (runtime api) error: ", func, " failed! ",
                          hipGetErrorString(result), " (", result, ") ",
                          common::get_file_from_filepath(file), ":", line));
  }

  return true;
}
}
}