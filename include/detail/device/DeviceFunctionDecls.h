//
// Created by mhaidl on 05/06/16.
//

#ifndef PACXX_V2_DEVICEFUNCTIONDECLS_H
#define PACXX_V2_DEVICEFUNCTIONDECLS_H
#pragma once


#ifdef __CUDA_DEVICE_CODE
using __index_t = unsigned int;
#else
using __index_t = unsigned long;
#endif

__index_t get_global_id(unsigned int dimindx);
__index_t get_local_id(unsigned int dimindx);
__index_t get_group_id(unsigned int dimindx);
__index_t get_local_size(unsigned int dimindx);
__index_t get_num_groups(unsigned int dimindx);
__index_t get_grid_size(unsigned int dimindx);


//#ifdef __CUDA_DEVICE_CODE
void barrier(unsigned int flags);
//#else
//void barrier(unsigned int flags) { __asm__ volatile("" : : : "memory"); }
//#endif


extern "C" double exp(double val);
extern "C" double log(double val);
extern "C" double sin(double val);
extern "C" double cos(double val);
extern "C" float expf(float val);
extern "C" float logf(float val);
extern "C" float sinf(float val);
extern "C" float cosf(float val);

// function declarations for the device functions provided by Nvidia's NVVM

extern "C" float __nv_fast_sinf(float x);

extern "C" float __nv_fast_cosf(float x);

extern "C" float __nv_fast_log2f(float x);

extern "C" float __nv_fast_tanf(float x);

extern "C" void __nv_fast_sincosf(float x, float *sptr, float *cptr);

extern "C" float __nv_fast_expf(float x);

extern "C" float __nv_fast_exp10f(float x);

extern "C" float __nv_fast_log10f(float x);

extern "C" float __nv_fast_logf(float x);

extern "C" float __nv_fast_powf(float x, float y);

extern "C" int __nv_hadd(int x, int y);

extern "C" int __nv_rhadd(int x, int y);

extern "C" unsigned int __nv_uhadd(unsigned int x, unsigned int y);

extern "C" unsigned int __nv_urhadd(unsigned int x, unsigned int y);

extern "C" float __nv_fsub_rn(float x, float y);

extern "C" float __nv_fsub_rz(float x, float y);

extern "C" float __nv_fsub_rd(float x, float y);

extern "C" float __nv_fsub_ru(float x, float y);

extern "C" float __nv_frsqrt_rn(float x);

extern "C" int __nv_ffs(int x);

extern "C" int __nv_ffsll(long long int x);

extern "C" float __nv_rintf(float x);

extern "C" long long int __nv_llrintf(float x);

extern "C" float __nv_nearbyintf(float x);

extern "C" int __nv_signbitf(float x);

extern "C" float __nv_copysignf(float x, float y);

extern "C" int __nv_finitef(float x);

extern "C" int __nv_isinff(float x);

extern "C" int __nv_isnanf(float x);

extern "C" float __nv_nextafterf(float x, float y);

extern "C" float __nv_nanf(const signed char *tagp);

extern "C" float __nv_sinf(float x);

extern "C" float __nv_cosf(float x);

extern "C" void __nv_sincosf(float x, float *sptr, float *cptr);

extern "C" float __nv_sinpif(float x);

extern "C" float __nv_cospif(float x);

extern "C" void __nv_sincospif(float x, float *sptr, float *cptr);

extern "C" float __nv_tanf(float x);

extern "C" float __nv_log2f(float x);

extern "C" float __nv_expf(float x);

extern "C" float __nv_exp10f(float x);

extern "C" float __nv_coshf(float x);

extern "C" float __nv_sinhf(float x);

extern "C" float __nv_tanhf(float x);

extern "C" float __nv_atan2f(float x, float y);

extern "C" float __nv_atanf(float x);

extern "C" float __nv_asinf(float x);

extern "C" float __nv_acosf(float x);

extern "C" float __nv_logf(float x);

extern "C" float __nv_log10f(float x);

extern "C" float __nv_log1pf(float x);

extern "C" float __nv_acoshf(float x);

extern "C" float __nv_asinhf(float x);

extern "C" float __nv_atanhf(float x);

extern "C" float __nv_expm1f(float x);

extern "C" float __nv_hypotf(float x, float y);


extern "C" float __nv_rhypotf(float x, float y);
extern "C" float __nv_cbrtf(float x);

extern "C" float __nv_rcbrtf(float x);

extern "C" float __nv_j0f(float x);

extern "C" float __nv_j1f(float x);

extern "C" float __nv_y0f(float x);

extern "C" float __nv_y1f(float x);

extern "C" float __nv_ynf(int n, float x);

extern "C" float __nv_jnf(int n, float x);

extern "C" float __nv_cyl_bessel_i0f(float x);

extern "C" float __nv_cyl_bessel_i1f(float x);

extern "C" float __nv_erff(float x);

extern "C" float __nv_erfinvf(float x);

extern "C" float __nv_erfcf(float x);

extern "C" float __nv_erfcxf(float x);

extern "C" float __nv_erfcinvf(float x);

extern "C" float __nv_normcdfinvf(float x);

extern "C" float __nv_normcdff(float x);

extern "C" float __nv_lgammaf(float x);

extern "C" float __nv_ldexpf(float x, int y);

extern "C" float __nv_scalbnf(float x, int y);

extern "C" float __nv_frexpf(float x, int *b);

extern "C" float __nv_modff(float x, float *b);

extern "C" float __nv_fmodf(float x, float y);

extern "C" float __nv_remainderf(float x, float y);

extern "C" float __nv_remquof(float x, float y, int *quo);

extern "C" float __nv_fmaf(float x, float y, float z);

extern "C" float __nv_powif(float x, int y);

extern "C" double __nv_powi(double x, int y);

extern "C" float __nv_powf(float x, float y);

extern "C" float __nv_tgammaf(float x);

extern "C" float __nv_roundf(float x);

extern "C" long long int __nv_llroundf(float x);

extern "C" float __nv_fdimf(float x, float y);

extern "C" int __nv_ilogbf(float x);

extern "C" float __nv_logbf(float x);

extern "C" double __nv_rint(double x);

extern "C" long long int __nv_llrint(double x);

extern "C" double __nv_nearbyint(double x);

extern "C" int __nv_signbitd(double x);

extern "C" int __nv_isfinited(double x);

extern "C" int __nv_isinfd(double x);

extern "C" int __nv_isnand(double x);

extern "C" double __nv_copysign(double x, double y);

extern "C" void __nv_sincos(double x, double *sptr, double *cptr);

extern "C" void __nv_sincospi(double x, double *sptr, double *cptr);

extern "C" double __nv_sin(double x);

extern "C" double __nv_cos(double x);

extern "C" double __nv_sinpi(double x);

extern "C" double __nv_cospi(double x);

extern "C" double __nv_tan(double x);

extern "C" double __nv_log(double x);

extern "C" double __nv_log2(double x);

extern "C" double __nv_log10(double x);

extern "C" double __nv_log1p(double x);

extern "C" double __nv_exp(double x);

extern "C" double __nv_exp2(double x);

extern "C" double __nv_exp10(double x);

extern "C" double __nv_expm1(double x);

extern "C" double __nv_cosh(double x);

extern "C" double __nv_sinh(double x);

extern "C" double __nv_tanh(double x);

extern "C" double __nv_atan2(double x, double y);

extern "C" double __nv_atan(double x);

extern "C" double __nv_asin(double x);

extern "C" double __nv_acos(double x);

extern "C" double __nv_acosh(double x);

extern "C" double __nv_asinh(double x);

extern "C" double __nv_atanh(double x);

extern "C" double __nv_hypot(double x, double y);

extern "C" double __nv_rhypot(double x, double y);
extern "C" double __nv_cbrt(double x);

extern "C" double __nv_rcbrt(double x);

extern "C" double __nv_pow(double x, double y);

extern "C" double __nv_j0(double x);

extern "C" double __nv_j1(double x);

extern "C" double __nv_y0(double x);

extern "C" double __nv_y1(double x);

extern "C" double __nv_yn(int n, double x);

extern "C" double __nv_jn(int n, double x);

extern "C" double __nv_cyl_bessel_i0(double x);

extern "C" double __nv_cyl_bessel_i1(double x);

extern "C" double __nv_erf(double x);

extern "C" double __nv_erfinv(double x);

extern "C" double __nv_erfcinv(double x);

extern "C" double __nv_normcdfinv(double x);

extern "C" double __nv_erfc(double x);

extern "C" double __nv_erfcx(double x);

extern "C" double __nv_normcdf(double x);

extern "C" double __nv_tgamma(double x);

extern "C" double __nv_lgamma(double x);

extern "C" double __nv_ldexp(double x, int y);

extern "C" double __nv_scalbn(double x, int y);

extern "C" double __nv_frexp(double x, int *b);

extern "C" double __nv_modf(double x, double *b);

extern "C" double __nv_fmod(double x, double y);

extern "C" double __nv_remainder(double x, double y);

extern "C" double __nv_remquo(double x, double y, int *c);

extern "C" double __nv_nextafter(double x, double y);

extern "C" double __nv_nan(const signed char *tagp);

extern "C" double __nv_round(double x);

extern "C" long long int __nv_llround(double x);

extern "C" double __nv_fdim(double x, double y);

extern "C" int __nv_ilogb(double x);

extern "C" double __nv_logb(double x);

extern "C" double __nv_fma(double x, double y, double z);

extern "C" int __nv_clz(int x);
extern "C" int __nv_clzll(long long x);

extern "C" int __nv_popc(int x);
extern "C" int __nv_popcll(long long x);

extern "C" unsigned int __nv_byte_perm(unsigned int x, unsigned int y,
                                       unsigned int z);

extern "C" int __nv_min(int x, int y);
extern "C" unsigned int __nv_umin(unsigned int x, unsigned int y);
extern "C" long long __nv_llmin(long long x, long long y);
extern "C" unsigned long long __nv_ullmin(unsigned long long x,
                                          unsigned long long y);

extern "C" int __nv_max(int x, int y);
extern "C" unsigned int __nv_umax(unsigned int x, unsigned int y);
extern "C" long long __nv_llmax(long long x, long long y);
extern "C" unsigned long long __nv_ullmax(unsigned long long x,
                                          unsigned long long y);

extern "C" int __nv_mulhi(int x, int y);
extern "C" unsigned int __nv_umulhi(unsigned int x, unsigned int y);
extern "C" long long __nv_mul64hi(long long x, long long y);
extern "C" unsigned long long __nv_umul64hi(unsigned long long x,
                                            unsigned long long y);

extern "C" int __nv_mul24(int x, int y);
extern "C" unsigned int __nv_umul24(unsigned int x, unsigned int y);

extern "C" unsigned int __nv_brev(unsigned int x);

extern "C" unsigned long long __nv_brevll(unsigned long long x);
extern "C" int __nv_sad(int x, int y, int z);

extern "C" unsigned int __nv_usad(unsigned int x, unsigned int y,
                                  unsigned int z);
extern "C" int __nv_abs(int x);

extern "C" long long __nv_llabs(long long x);

extern "C" float __nv_floorf(float f);

extern "C" double __nv_floor(double f);

extern "C" float __nv_fabsf(float f);

extern "C" double __nv_fabs(double f);

extern "C" double __nv_rcp64h(double d);

extern "C" float __nv_fminf(float x, float y);

extern "C" float __nv_fmaxf(float x, float y);

extern "C" float __nv_rsqrtf(float x);

extern "C" double __nv_fmin(double x, double y);

extern "C" double __nv_fmax(double x, double y);

extern "C" double __nv_rsqrt(double x);

extern "C" double __nv_ceil(double x);

extern "C" double __nv_trunc(double x);

extern "C" float __nv_exp2f(float x);

extern "C" float __nv_truncf(float x);

extern "C" float __nv_ceilf(float x);

extern "C" float __nv_saturatef(float x);

extern "C" float __nv_fmaf_rn(float x, float y, float z);
extern "C" float __nv_fmaf_rz(float x, float y, float z);
extern "C" float __nv_fmaf_rd(float x, float y, float z);
extern "C" float __nv_fmaf_ru(float x, float y, float z);

extern "C" float
    __nv_fmaf_ieee_rn(float x, float y, float z);
extern "C" float
    __nv_fmaf_ieee_rz(float x, float y, float z);
extern "C" float
    __nv_fmaf_ieee_rd(float x, float y, float z);
extern "C" float
    __nv_fmaf_ieee_ru(float x, float y, float z);

extern "C" double __nv_fma_rn(double x, double y, double z);
extern "C" double __nv_fma_rz(double x, double y, double z);
extern "C" double __nv_fma_rd(double x, double y, double z);
extern "C" double __nv_fma_ru(double x, double y, double z);

extern "C" float __nv_fast_fdividef(float x, float y);

extern "C" float __nv_fdiv_rn(float x, float y);
extern "C" float __nv_fdiv_rz(float x, float y);
extern "C" float __nv_fdiv_rd(float x, float y);
extern "C" float __nv_fdiv_ru(float x, float y);

extern "C" float __nv_frcp_rn(float x);
extern "C" float __nv_frcp_rz(float x);
extern "C" float __nv_frcp_rd(float x);
extern "C" float __nv_frcp_ru(float x);

extern "C" float __nv_fsqrt_rn(float x);
extern "C" float __nv_fsqrt_rz(float x);
extern "C" float __nv_fsqrt_rd(float x);
extern "C" float __nv_fsqrt_ru(float x);

extern "C" double __nv_ddiv_rn(double x, double y);
extern "C" double __nv_ddiv_rz(double x, double y);
extern "C" double __nv_ddiv_rd(double x, double y);
extern "C" double __nv_ddiv_ru(double x, double y);

extern "C" double __nv_drcp_rn(double x);
extern "C" double __nv_drcp_rz(double x);
extern "C" double __nv_drcp_rd(double x);
extern "C" double __nv_drcp_ru(double x);

extern "C" double __nv_dsqrt_rn(double x);

extern "C" double __nv_dsqrt_rz(double x);
extern "C" double __nv_dsqrt_rd(double x);
extern "C" double __nv_dsqrt_ru(double x);

extern "C" float __nv_sqrtf(float x);

extern "C" double __nv_sqrt(double x);

extern "C" double __nv_dadd_rn(double x, double y);
extern "C" double __nv_dadd_rz(double x, double y);
extern "C" double __nv_dadd_rd(double x, double y);
extern "C" double __nv_dadd_ru(double x, double y);

extern "C" double __nv_dmul_rn(double x, double y);
extern "C" double __nv_dmul_rz(double x, double y);
extern "C" double __nv_dmul_rd(double x, double y);
extern "C" double __nv_dmul_ru(double x, double y);

extern "C" float __nv_fadd_rd(float x, float y);
extern "C" float __nv_fadd_ru(float x, float y);

extern "C" float __nv_fmul_rd(float x, float y);
extern "C" float __nv_fmul_ru(float x, float y);

extern "C" float __nv_fadd_rn(float x, float y);
extern "C" float __nv_fadd_rz(float x, float y);

extern "C" float __nv_fmul_rn(float x, float y);
extern "C" float __nv_fmul_rz(float x, float y);

extern "C" float __nv_double2float_rn(double d);
extern "C" float __nv_double2float_rz(double d);
extern "C" float __nv_double2float_rd(double d);
extern "C" float __nv_double2float_ru(double d);

extern "C" int __nv_double2int_rn(double d);
extern "C" int __nv_double2int_rz(double d);
extern "C" int __nv_double2int_rd(double d);
extern "C" int __nv_double2int_ru(double d);

extern "C" unsigned int __nv_double2uint_rn(double d);
extern "C" unsigned int __nv_double2uint_rz(double d);
extern "C" unsigned int __nv_double2uint_rd(double d);
extern "C" unsigned int __nv_double2uint_ru(double d);

extern "C" double __nv_int2double_rn(int i);

extern "C" double __nv_uint2double_rn(unsigned int i);

extern "C" int __nv_float2int_rn(float in);
extern "C" int __nv_float2int_rz(float in);
extern "C" int __nv_float2int_rd(float in);
extern "C" int __nv_float2int_ru(float in);
extern "C" unsigned int __nv_float2uint_rn(float in);
extern "C" unsigned int __nv_float2uint_rz(float in);
extern "C" unsigned int __nv_float2uint_rd(float in);
extern "C" unsigned int __nv_float2uint_ru(float in);

extern "C" float __nv_int2float_rn(int in);
extern "C" float __nv_int2float_rz(int in);
extern "C" float __nv_int2float_rd(int in);
extern "C" float __nv_int2float_ru(int in);

extern "C" float __nv_uint2float_rn(unsigned int in);
extern "C" float __nv_uint2float_rz(unsigned int in);
extern "C" float __nv_uint2float_rd(unsigned int in);
extern "C" float __nv_uint2float_ru(unsigned int in);

extern "C" double __nv_hiloint2double(int x, int y);
extern "C" int __nv_double2loint(double d);
extern "C" int __nv_double2hiint(double d);

extern "C" long long __nv_float2ll_rn(float f);
extern "C" long long __nv_float2ll_rz(float f);
extern "C" long long __nv_float2ll_rd(float f);
extern "C" long long __nv_float2ll_ru(float f);
extern "C" unsigned long long __nv_float2ull_rn(float f);
extern "C" unsigned long long __nv_float2ull_rz(float f);
extern "C" unsigned long long __nv_float2ull_rd(float f);
extern "C" unsigned long long __nv_float2ull_ru(float f);

extern "C" long long __nv_double2ll_rn(double f);
extern "C" long long __nv_double2ll_rz(double f);
extern "C" long long __nv_double2ll_rd(double f);
extern "C" long long __nv_double2ll_ru(double f);

extern "C" unsigned long long __nv_double2ull_rn(double f);
extern "C" unsigned long long __nv_double2ull_rz(double f);
extern "C" unsigned long long __nv_double2ull_rd(double f);
extern "C" unsigned long long __nv_double2ull_ru(double f);

extern "C" float __nv_ll2float_rn(long long l);
extern "C" float __nv_ll2float_rz(long long l);
extern "C" float __nv_ll2float_rd(long long l);
extern "C" float __nv_ll2float_ru(long long l);

extern "C" float __nv_ull2float_rn(unsigned long long l);
extern "C" float __nv_ull2float_rz(unsigned long long l);
extern "C" float __nv_ull2float_rd(unsigned long long l);
extern "C" float __nv_ull2float_ru(unsigned long long l);

extern "C" double __nv_ll2double_rn(long long l);
extern "C" double __nv_ll2double_rz(long long l);
extern "C" double __nv_ll2double_rd(long long l);
extern "C" double __nv_ll2double_ru(long long l);

extern "C" double __nv_ull2double_rn(unsigned long long l);
extern "C" double __nv_ull2double_rz(unsigned long long l);
extern "C" double __nv_ull2double_rd(unsigned long long l);
extern "C" double __nv_ull2double_ru(unsigned long long l);

extern "C" unsigned short __nv_float2half_rn(float f);
extern "C" float __nv_half2float(unsigned short h);

extern "C" float __nv_int_as_float(int x);
extern "C" float __nv_uint_as_float(unsigned int x);

extern "C" int __nv_float_as_int(float x);
extern "C" unsigned int __nv_float_as_uint(float x);

extern "C" double __nv_longlong_as_double(long long x);
extern "C" long long __nv_double_as_longlong(double x);

#endif //PACXX_V2_DEVICEFUNCTIONDECLS_H
