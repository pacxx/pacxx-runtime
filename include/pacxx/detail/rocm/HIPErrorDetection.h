//
// Created by mhaidl on 30/05/16.
//

#ifndef PACXX_V2_HIPERRORDETECTION_H
#define PACXX_V2_HIPERRORDETECTION_H

#define __HIP_PLATFORM_HCC__
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
namespace pacxx {
namespace v2 {

bool checkHIPCall(hipError_t result, char const *const func,
                   const char *const file, int const line);
}
}

#define SEC_HIP_CALL(val)                                                     \
  pacxx::v2::checkHIPCall((val), #val, __FILE__, __LINE__)

#endif // PACXX_V2_HIPERRORDETECTION_H
