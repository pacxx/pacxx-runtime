//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef PACXX_V2_CUDAERRORDETECTION_H
#define PACXX_V2_CUDAERRORDETECTION_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cupti.h>
namespace pacxx {
namespace v2 {
bool checkCUDACall(CUresult result, char const *const func,
                   const char *const file, int const line);

bool checkCUDACall(cudaError_t result, char const *const func,
                   const char *const file, int const line);

bool checkCUPTICall(CUptiResult result, char const *const func,
                   const char *const file, int const line);
}
}

#define SEC_CUDA_CALL(val)                                                     \
  pacxx::v2::checkCUDACall((val), #val, __FILE__, __LINE__)
#define SEC_CUPTI_CALL(val)                                                     \
  pacxx::v2::checkCUPTICall((val), #val, __FILE__, __LINE__)

#endif // PACXX_V2_CUDAERRORDETECTION_H
