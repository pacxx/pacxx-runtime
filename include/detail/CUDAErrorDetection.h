//
// Created by mhaidl on 30/05/16.
//

#ifndef PACXX_V2_CUDAERRORDETECTION_H
#define PACXX_V2_CUDAERRORDETECTION_H

#include <cuda.h>
#include <cuda_runtime.h>

namespace pacxx
{
  namespace v2
  {
    bool checkCUDACall(CUresult result, char const *const func,
                   const char *const file, int const line);

    bool checkCUDACall(cudaError_t result, char const *const func,
                   const char *const file, int const line);

  }
}

#define SEC_CUDA_CALL(val) checkCUDACall((val), #val, __FILE__, __LINE__)


#endif //PACXX_V2_CUDAERRORDETECTION_H
