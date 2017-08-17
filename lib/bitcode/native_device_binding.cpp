#include </usr/include/math.h> // FIXME: PACXX gets confused with cmath from libc++ on some platforms thats
                               //        why we include math.h with an absolute path here

#ifndef __forceinline__
#define __forceinline__ __attribute__((always_inline))
#endif

extern "C" __forceinline__ double rsqrt(double val) {
  return 1.0 / sqrt(val);
}

extern "C" __forceinline__ float rsqrtf(float val) {
  return 1.0f / sqrtf(val);
}