//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#pragma once
//#include <device_types.h>
#ifndef __forceinline__
#define __forceinline__ __attribute__((always_inline))
#endif
#include "DevicePrintf.h"
#include "DeviceTypes.h"

// atomics
#define __atomic_op_decl_(op, ctype, memory)                                   \
  __forceinline__ ctype atomic_##op(memory##_mem_##ctype##_ptr ptr,            \
                                    ctype value);

#define atomic_decl_(op, ctype)                                                \
  __atomic_op_decl_(op, ctype, global) __atomic_op_decl_(op, ctype, shared)

atomic_decl_(add, int) 
atomic_decl_(add, uint) 
atomic_decl_(add, ulong)
atomic_decl_(add, float)

atomic_decl_(min, int) 
atomic_decl_(min, uint) 
atomic_decl_(min, long)
atomic_decl_(min, ulong)

atomic_decl_(max, int) 
atomic_decl_(max, uint)
atomic_decl_(max, long) 
atomic_decl_(max, ulong)

atomic_decl_(inc, uint) 
atomic_decl_(dec, uint)

atomic_decl_(cas, uint) 
atomic_decl_(cas, ulong)

atomic_decl_(exch, uint) 
atomic_decl_(exch,ulong)

atomic_decl_(and, uint) 
atomic_decl_(and, ulong)
atomic_decl_(or, uint)
atomic_decl_(or, ulong)
atomic_decl_(xor, uint)
atomic_decl_(xor, ulong)

namespace native {
  namespace index {
  struct idx {
    enum value { thread = 0, block = 1, global = 2 };
  };

  struct dim {
    enum value { block = 10, grid = 11 };
  };

  template <int> unsigned int x();
  template <int> unsigned int y();
  template <int> unsigned int z();
  } // namespace index
} /* native */

// generated intrinsic wrapper
namespace native {
namespace nvvm {
int clz(int val);
long long clz(long long val);
int popc(int val);
long long popc(long long val);
int prmt(int val1, int val2, int val3);
int min(int val1, int val2);
unsigned int min(unsigned int val1, unsigned int val2);
long long min(long long val1, long long val2);
unsigned long long min(unsigned long long val1, unsigned long long val2);
int max(int val1, int val2);
unsigned int max(unsigned int val1, unsigned int val2);
long long max(long long val1, long long val2);
unsigned long long max(unsigned long long val1, unsigned long long val2);
float fmin(float val1, float val2);
float fmin_ftz(float val1, float val2);
float fmax(float val1, float val2);
float fmax_ftz(float val1, float val2);
double fmin(double val1, double val2);
double fmax(double val1, double val2);
int mulhi(int val1, int val2);
unsigned int mulhi(unsigned int val1, unsigned int val2);
long long mulhi(long long val1, long long val2);
unsigned long long mulhi(unsigned long long val1, unsigned long long val2);
float mul_rn_ftz(float val1, float val2);
float mul_rn(float val1, float val2);
float mul_rz_ftz(float val1, float val2);
float mul_rz(float val1, float val2);
float mul_rm_ftz(float val1, float val2);
float mul_rm(float val1, float val2);
float mul_rp_ftz(float val1, float val2);
float mul_rp(float val1, float val2);
double mul_rn(double val1, double val2);
double mul_rz(double val1, double val2);
double mul_rm(double val1, double val2);
double mul_rp(double val1, double val2);
int mul24(int val1, int val2);
unsigned int mul24(unsigned int val1, unsigned int val2);
float div_ftz(float val1, float val2);
float div(float val1, float val2);
float div_rn_ftz(float val1, float val2);
float div_rn(float val1, float val2);
float div_rz_ftz(float val1, float val2);
float div_rz(float val1, float val2);
float div_rm_ftz(float val1, float val2);
float div_rm(float val1, float val2);
float div_rp_ftz(float val1, float val2);
float div_rp(float val1, float val2);
double div_rn(double val1, double val2);
double div_rz(double val1, double val2);
double div_rm(double val1, double val2);
double div_rp(double val1, double val2);
int brev32(int val);
long brev64(long val);
int sad(int val1, int val2, int val3);
unsigned int sad(unsigned int val1, unsigned int val2, unsigned int val3);
float floor_ftz(float val);
float floor(float val);
double floor(double val);
float ceil_ftz(float val);
float ceil(float val);
double ceil(double val);
int abs(int val);
long long abs(long long val);
float fabs_ftz(float val);
float fabs(float val);
double fabs(double val);
float round_ftz(float val);
float round(float val);
double round(double val);
float trunc_ftz(float val);
float trunc(float val);
double trunc(double val);
float saturate_ftz(float val);
float saturate(float val);
double saturate(double val);
float ex2_ftz(float val);
float ex2(float val);
double ex2(double val);
float lg2_ftz(float val);
float lg2(float val);
double lg2(double val);
float sin_ftz(float val);
float sin(float val);
float cos_ftz(float val);
float cos(float val);
float fma_rn_ftz(float val1, float val2, float val3);
float fma_rn(float val1, float val2, float val3);
float fma_rz_ftz(float val1, float val2, float val3);
float fma_rz(float val1, float val2, float val3);
float fma_rm_ftz(float val1, float val2, float val3);
float fma_rm(float val1, float val2, float val3);
float fma_rp_ftz(float val1, float val2, float val3);
float fma_rp(float val1, float val2, float val3);
double fma_rn(double val1, double val2, double val3);
double fma_rz(double val1, double val2, double val3);
double fma_rm(double val1, double val2, double val3);
double fma_rp(double val1, double val2, double val3);
float rcp_rn_ftz(float val);
float rcp_rn(float val);
float rcp_rz_ftz(float val);
float rcp_rz(float val);
float rcp_rm_ftz(float val);
float rcp_rm(float val);
float rcp_rp_ftz(float val);
float rcp_rp(float val);
double rcp_rn(double val);
double rcp_rz(double val);
double rcp_rm(double val);
double rcp_rp(double val);
double rcp_ftz(double val);
float sqrt_rn_ftz(float val);
float sqrt_rn(float val);
float sqrt_rz_ftz(float val);
float sqrt_rz(float val);
float sqrt_rm_ftz(float val);
float sqrt_rm(float val);
float sqrt_rp_ftz(float val);
float sqrt_rp(float val);
float sqrt_ftz(float val);
float sqrt_ap(float val);
double sqrt_rn(double val);
double sqrt_rz(double val);
double sqrt_rm(double val);
double sqrt_rp(double val);
float rsqrt_ftz(float val);
float rsqrt(float val);
double rsqrt(double val);
float add_rn_ftz(float val1, float val2);
float add_rn(float val1, float val2);
float add_rz_ftz(float val1, float val2);
float add_rz(float val1, float val2);
float add_rm_ftz(float val1, float val2);
float add_rm(float val1, float val2);
float add_rp_ftz(float val1, float val2);
float add_rp(float val1, float val2);
double add_rn(double val1, double val2);
double add_rz(double val1, double val2);
double add_rm(double val1, double val2);
double add_rp(double val1, double val2);
float d2f_rn_ftz(double val);
float d2f_rn(double val);
float d2f_rz_ftz(double val);
float d2f_rz(double val);
float d2f_rm_ftz(double val);
float d2f_rm(double val);
float d2f_rp_ftz(double val);
float d2f_rp(double val);
int d2i_rn(double val);
int d2i_rz(double val);
int d2i_rm(double val);
int d2i_rp(double val);
unsigned int d2ui_rn(double val);
unsigned int d2ui_rz(double val);
unsigned int d2ui_rm(double val);
unsigned int d2ui_rp(double val);
double i2d_rn(int val);
double i2d_rz(int val);
double i2d_rm(int val);
double i2d_rp(int val);
double ui2d_rn(unsigned int val);
double ui2d_rz(unsigned int val);
double ui2d_rm(unsigned int val);
double ui2d_rp(unsigned int val);
int f2i_rn_ftz(float val);
int f2i_rn(float val);
int f2i_rz_ftz(float val);
int f2i_rz(float val);
int f2i_rm_ftz(float val);
int f2i_rm(float val);
int f2i_rp_ftz(float val);
int f2i_rp(float val);
unsigned int f2ui_rn_ftz(float val);
unsigned int f2ui_rn(float val);
unsigned int f2ui_rz_ftz(float val);
unsigned int f2ui_rz(float val);
unsigned int f2ui_rm_ftz(float val);
unsigned int f2ui_rm(float val);
unsigned int f2ui_rp_ftz(float val);
unsigned int f2ui_rp(float val);
float i2f_rn(int val);
float i2f_rz(int val);
float i2f_rm(int val);
float i2f_rp(int val);
float ui2f_rn(unsigned int val);
float ui2f_rz(unsigned int val);
float ui2f_rm(unsigned int val);
float ui2f_rp(unsigned int val);
double lohi_i2d(int val1, int val2);
int d2i_lo(double val);
int d2i_hi(double val);
long long f2ll_rn_ftz(float val);
long long f2ll_rn(float val);
long long f2ll_rz_ftz(float val);
long long f2ll_rz(float val);
long long f2ll_rm_ftz(float val);
long long f2ll_rm(float val);
long long f2ll_rp_ftz(float val);
long long f2ll_rp(float val);
unsigned long long f2ull_rn_ftz(float val);
unsigned long long f2ull_rn(float val);
unsigned long long f2ull_rz_ftz(float val);
unsigned long long f2ull_rz(float val);
unsigned long long f2ull_rm_ftz(float val);
unsigned long long f2ull_rm(float val);
unsigned long long f2ull_rp_ftz(float val);
unsigned long long f2ull_rp(float val);
long long d2ll_rn(double val);
long long d2ll_rz(double val);
long long d2ll_rm(double val);
long long d2ll_rp(double val);
unsigned long long d2ull_rn(double val);
unsigned long long d2ull_rz(double val);
unsigned long long d2ull_rm(double val);
unsigned long long d2ull_rp(double val);
float ll2f_rn(long long val);
float ll2f_rz(long long val);
float ll2f_rm(long long val);
float ll2f_rp(long long val);
float ull2f_rn(unsigned long long val);
float ull2f_rz(unsigned long long val);
float ull2f_rm(unsigned long long val);
float ull2f_rp(unsigned long long val);
double ll2d_rn(long long val);
double ll2d_rz(long long val);
double ll2d_rm(long long val);
double ll2d_rp(long long val);
double ull2d_rn(unsigned long long val);
double ull2d_rz(unsigned long long val);
double ull2d_rm(unsigned long long val);
double ull2d_rp(unsigned long long val);
short f2h_rn_ftz(float val);
short f2h_rn(float val);
float h2f(short val);
int bitcast_f2i(float val);
float bitcast_i2f(int val);
double bitcast_ll2d(long long val);
long long bitcast_d2ll(double val);
} // namespace nvvm

float exp(float val);
float log(float val);
float sqrt(float val);
float rsqrt(float val);
float fabs(float val);
float sin(float val);
float cos(float val);
// barriers
void barrier();
/*
namespace atomic {
namespace global {
template <typename T> T atomic_add(T *ptr, T value) {
#ifdef __device_code__
  return ::atomic_add((volatile __attribute__((address_space(1))) T *)ptr,
                      value);
#else
  return T();
#endif
}

template <typename T> T atomic_and(T *ptr, T value) {
#ifdef __device_code__
  return ::atomic_and((volatile __attribute__((address_space(1))) T *)ptr,
                      value);
#endif
}

template <typename T> T atomic_or(T *ptr, T value) {
#ifdef __device_code__
  return ::atomic_or((volatile __attribute__((address_space(1))) T *)ptr,
                     value);
#else
  return T();
#endif
}

template <typename T> T atomic_xor(T *ptr, T value) {
#ifdef __device_code__
  return ::atomic_xor((volatile __attribute__((address_space(1))) T *)ptr,
                      value);
#else
  return T();
#endif
}
template <typename T> T atomic_max(T *ptr, T value) {
#ifdef __device_code__
  return ::atomic_max((volatile __attribute__((address_space(1))) T *)ptr,
                      value);
#else
  return T();
#endif
}

template <typename T> T atomic_min(T *ptr, T value) {
#ifdef __device_code__
  return ::atomic_min((volatile __attribute__((address_space(1))) T *)ptr,
                      value);
#else
  return T();
#endif
}
}

namespace shared {
template <typename T> T atomic_add(T *ptr, T value) {
#ifdef __device_code__
  return ::atomic_add((volatile __attribute__((address_space(3))) T *)ptr,
                      value);
#else
  return T();
#endif
}

template <typename T> T atomic_and(T *ptr, T value) {
#ifdef __device_code__
  return ::atomic_and((volatile __attribute__((address_space(3))) T *)ptr,
                      value);
#else
  return T();
#endif
}

template <typename T> T atomic_or(T *ptr, T value) {
#ifdef __device_code__
  return ::atomic_or((volatile __attribute__((address_space(3))) T *)ptr,
                     value);
#else
  return T();
#endif
}

template <typename T> T atomic_xor(T *ptr, T value) {
#ifdef __device_code__
  return ::atomic_xor((volatile __attribute__((address_space(3))) T *)ptr,
                      value);
#else
  return T();
#endif
}

template <typename T> T atomic_max(T *ptr, T value) {
#ifdef __device_code__
  return ::atomic_max((volatile __attribute__((address_space(3))) T *)ptr,
                      value);
#else
  return T();
#endif
}

template <typename T> T atomic_min(T *ptr, T value) {
#ifdef __device_code__
  return ::atomic_min((volatile __attribute__((address_space(3))) T *)ptr,
                      value);
#else
  return T();
#endif
}
}

}
*/
} // namespace native

#ifdef signbit
#undef signbit
#endif
int signbit(double val);

#ifndef __device_code__
#define __syncthreads(...)
#else
//#define __syncthreads(...) native::barrier()
#endif

#ifdef __device_code__
#ifdef __PACXX_CUDA_MODE__

#include <math.h>

#endif
#endif
