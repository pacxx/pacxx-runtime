//
// Created by mhaidl on 05/06/16.
//

/* Copyright (C) University of Muenster - All Rights Reserved
* Unauthorized copying of this file, via any medium is strictly prohibited
* Proprietary and confidential
* Written by Michael Haidl <michael.haidl@uni-muenster.de>, 2010-2014
*/

#include "detail/device/DeviceCode.h"
#ifdef __device_code__

#define __CUDANVVM__ 1
#define __CUDABE__ 1
//#define __CUDACC_RTC__ 1
//#include <vector_types.h>
#include "detail/device/DeviceFunctionDecls.h"
extern "C"{
//#include <device_functions_decls.h>
int __nvvm_atom_cta_add_gen_i(volatile int *, int);
int __nvvm_atom_sys_add_gen_i(volatile int *, int);
long long __nvvm_atom_cta_add_gen_ll(volatile long long *, long long);
long long __nvvm_atom_sys_add_gen_ll(volatile long long *, long long);
unsigned int __nvvm_atom_cta_add_gen_ui(volatile unsigned int *, unsigned int);
unsigned int __nvvm_atom_sys_add_gen_ui(volatile unsigned int *, unsigned int);
unsigned long long __nvvm_atom_cta_add_gen_ull(volatile unsigned long long *, unsigned long long);
unsigned long long __nvvm_atom_sys_add_gen_ull(volatile unsigned long long *, unsigned long long);
float __nvvm_atom_cta_add_gen_f(volatile float*, float); 
float __nvvm_atom_sys_add_gen_f(volatile float*, float); 
double __nvvm_atom_cta_add_gen_d(volatile double*, double); 
double __nvvm_atom_sys_add_gen_d(volatile double*, double); 

int __nvvm_atom_cta_min_gen_i(volatile int *, int);
int __nvvm_atom_sys_min_gen_i(volatile int *, int);
long long __nvvm_atom_cta_min_gen_ll(volatile long long *, long long);
long long __nvvm_atom_sys_min_gen_ll(volatile long long *, long long);
unsigned int __nvvm_atom_cta_min_gen_ui(volatile unsigned int *, unsigned int);
unsigned int __nvvm_atom_sys_min_gen_ui(volatile unsigned int *, unsigned int);
unsigned long long __nvvm_atom_cta_min_gen_ull(volatile unsigned long long *, unsigned long long);
unsigned long long __nvvm_atom_sys_min_gen_ull(volatile unsigned long long *, unsigned long long);
float __nvvm_atom_cta_min_gen_f(volatile float*, float); 
float __nvvm_atom_sys_min_gen_f(volatile float*, float); 
double __nvvm_atom_cta_min_gen_d(volatile double*, double); 
double __nvvm_atom_sys_min_gen_d(volatile double*, double); 

int __nvvm_atom_cta_max_gen_i(volatile int *, int);
int __nvvm_atom_sys_max_gen_i(volatile int *, int);
long long __nvvm_atom_cta_max_gen_ll(volatile long long *, long long);
long long __nvvm_atom_sys_max_gen_ll(volatile long long *, long long);
unsigned int __nvvm_atom_cta_max_gen_ui(volatile unsigned int *, unsigned int);
unsigned int __nvvm_atom_sys_max_gen_ui(volatile unsigned int *, unsigned int);
unsigned long long __nvvm_atom_cta_max_gen_ull(volatile unsigned long long *, unsigned long long);
unsigned long long __nvvm_atom_sys_max_gen_ull(volatile unsigned long long *, unsigned long long);
float __nvvm_atom_cta_max_gen_f(volatile float*, float); 
float __nvvm_atom_sys_max_gen_f(volatile float*, float); 
double __nvvm_atom_cta_max_gen_d(volatile double*, double); 
double __nvvm_atom_sys_max_gen_d(volatile double*, double); 

int __nvvm_atom_cta_xchg_gen_i(volatile int *, int);
int __nvvm_atom_sys_xchg_gen_i(volatile int *, int);
long long __nvvm_atom_cta_xchg_gen_ll(volatile long long *, long long);
long long __nvvm_atom_sys_xchg_gen_ll(volatile long long *, long long);
unsigned int __nvvm_atom_cta_xchg_gen_ui(volatile unsigned int *, unsigned int);
unsigned int __nvvm_atom_sys_xchg_gen_ui(volatile unsigned int *, unsigned int);
unsigned long long __nvvm_atom_cta_xchg_gen_ull(volatile unsigned long long *, unsigned long long);
unsigned long long __nvvm_atom_sys_xchg_gen_ull(volatile unsigned long long *, unsigned long long);
float __nvvm_atom_cta_xchg_gen_f(volatile float*, float); 
float __nvvm_atom_sys_xchg_gen_f(volatile float*, float); 
double __nvvm_atom_cta_xchg_gen_d(volatile double*, double); 
double __nvvm_atom_sys_xchg_gen_d(volatile double*, double); 

int __nvvm_atom_cta_inc_gen_i(volatile int *, int);
int __nvvm_atom_sys_inc_gen_i(volatile int *, int);
long long __nvvm_atom_cta_inc_gen_ll(volatile long long *, long long);
long long __nvvm_atom_sys_inc_gen_ll(volatile long long *, long long);
unsigned int __nvvm_atom_cta_inc_gen_ui(volatile unsigned int *, unsigned int);
unsigned int __nvvm_atom_sys_inc_gen_ui(volatile unsigned int *, unsigned int);
unsigned long long __nvvm_atom_cta_inc_gen_ull(volatile unsigned long long *, unsigned long long);
unsigned long long __nvvm_atom_sys_inc_gen_ull(volatile unsigned long long *, unsigned long long);

int __nvvm_atom_cta_dec_gen_i(volatile int *, int);
int __nvvm_atom_sys_dec_gen_i(volatile int *, int);
long long __nvvm_atom_cta_dec_gen_ll(volatile long long *, long long);
long long __nvvm_atom_sys_dec_gen_ll(volatile long long *, long long);
unsigned int __nvvm_atom_cta_dec_gen_ui(volatile unsigned int *, unsigned int);
unsigned int __nvvm_atom_sys_dec_gen_ui(volatile unsigned int *, unsigned int);
unsigned long long __nvvm_atom_cta_dec_gen_ull(volatile unsigned long long *, unsigned long long);
unsigned long long __nvvm_atom_sys_dec_gen_ull(volatile unsigned long long *, unsigned long long);

int __nvvm_atom_cta_cas_gen_i(volatile int *, int, int);
int __nvvm_atom_sys_cas_gen_i(volatile int *, int, int);
long long __nvvm_atom_cta_cas_gen_ll(volatile long long *, long long, long long);
long long __nvvm_atom_sys_cas_gen_ll(volatile long long *, long long, long long);

int __nvvm_atom_cta_and_gen_i(volatile int *, int);
int __nvvm_atom_sys_and_gen_i(volatile int *, int);
long long __nvvm_atom_cta_and_gen_ll(volatile long long *, long long);
long long __nvvm_atom_sys_and_gen_ll(volatile long long *, long long);

int __nvvm_atom_cta_or_gen_i(volatile int *, int);
int __nvvm_atom_sys_or_gen_i(volatile int *, int);
long long __nvvm_atom_cta_or_gen_ll(volatile long long *, long long);
long long __nvvm_atom_sys_or_gen_ll(volatile long long *, long long);

int __nvvm_atom_cta_xor_gen_i(volatile int *, int);
int __nvvm_atom_sys_xor_gen_i(volatile int *, int);
long long __nvvm_atom_cta_xor_gen_ll(volatile long long *, long long);
long long __nvvm_atom_sys_xor_gen_ll(volatile long long *, long long);



//#include <device_functions.h>
};
namespace native {
namespace index {
template <> unsigned int x<idx::thread>() { return get_local_id(0); }
template <> unsigned int y<idx::thread>() { return get_local_id(1); }
template <> unsigned int z<idx::thread>() { return get_local_id(2); }

template <> unsigned int x<dim::block>() { return get_local_size(0); }
template <> unsigned int y<dim::block>() { return get_local_size(1); }
template <> unsigned int z<dim::block>() { return get_local_size(2); }

template <> unsigned int x<idx::block>() { return get_group_id(0); }
template <> unsigned int y<idx::block>() { return get_group_id(1); }
template <> unsigned int z<idx::block>() { return get_group_id(2); }

template <> unsigned int x<dim::grid>() { return get_num_groups(0); }
template <> unsigned int y<dim::grid>() { return get_num_groups(1); }
template <> unsigned int z<dim::grid>() { return get_num_groups(2); }

template <> unsigned int x<idx::global>() { return get_global_id(0); }
template <> unsigned int y<idx::global>() { return get_global_id(1); }
template <> unsigned int z<idx::global>() { return get_global_id(2); }
}
}

namespace native {
/*namespace nvvm {
int clz(int val) { return __nvvm_clz_i(val); }
long long clz(long long val) { return __nvvm_clz_ll(val); }
int popc(int val) { return __nvvm_popc_i(val); }
long long popc(long long val) { return __nvvm_popc_ll(val); }
int prmt(int val1, int val2, int val3) { return __nvvm_prmt(val1, val2, val3); }
int min(int val1, int val2) { return __nvvm_min_i(val1, val2); }
unsigned int min(unsigned int val1, unsigned int val2) {
  return __nvvm_min_ui(val1, val2);
}
long long min(long long val1, long long val2) {
  return __nvvm_min_ll(val1, val2);
}
unsigned long long min(unsigned long long val1, unsigned long long val2) {
  return __nvvm_min_ull(val1, val2);
}
int max(int val1, int val2) { return __nvvm_max_i(val1, val2); }
unsigned int max(unsigned int val1, unsigned int val2) {
  return __nvvm_max_ui(val1, val2);
}
long long max(long long val1, long long val2) {
  return __nvvm_max_ll(val1, val2);
}
unsigned long long max(unsigned long long val1, unsigned long long val2) {
  return __nvvm_max_ull(val1, val2);
}
float fmin(float val1, float val2) { return __nvvm_fmin_f(val1, val2); }
float fmin_ftz(float val1, float val2) { return __nvvm_fmin_ftz_f(val1, val2); }
float fmax(float val1, float val2) { return __nvvm_fmax_f(val1, val2); }
float fmax_ftz(float val1, float val2) { return __nvvm_fmax_ftz_f(val1, val2); }
double fmin(double val1, double val2) { return __nvvm_fmin_d(val1, val2); }
double fmax(double val1, double val2) { return __nvvm_fmax_d(val1, val2); }
int mulhi(int val1, int val2) { return __nvvm_mulhi_i(val1, val2); }
unsigned int mulhi(unsigned int val1, unsigned int val2) {
  return __nvvm_mulhi_ui(val1, val2);
}
long long mulhi(long long val1, long long val2) {
  return __nvvm_mulhi_ll(val1, val2);
}
unsigned long long mulhi(unsigned long long val1, unsigned long long val2) {
  return __nvvm_mulhi_ull(val1, val2);
}
float mul_rn_ftz(float val1, float val2) {
  return __nvvm_mul_rn_ftz_f(val1, val2);
}
float mul_rn(float val1, float val2) { return __nvvm_mul_rn_f(val1, val2); }
float mul_rz_ftz(float val1, float val2) {
  return __nvvm_mul_rz_ftz_f(val1, val2);
}
float mul_rz(float val1, float val2) { return __nvvm_mul_rz_f(val1, val2); }
float mul_rm_ftz(float val1, float val2) {
  return __nvvm_mul_rm_ftz_f(val1, val2);
}
float mul_rm(float val1, float val2) { return __nvvm_mul_rm_f(val1, val2); }
float mul_rp_ftz(float val1, float val2) {
  return __nvvm_mul_rp_ftz_f(val1, val2);
}
float mul_rp(float val1, float val2) { return __nvvm_mul_rp_f(val1, val2); }
double mul_rn(double val1, double val2) { return __nvvm_mul_rn_d(val1, val2); }
double mul_rz(double val1, double val2) { return __nvvm_mul_rz_d(val1, val2); }
double mul_rm(double val1, double val2) { return __nvvm_mul_rm_d(val1, val2); }
double mul_rp(double val1, double val2) { return __nvvm_mul_rp_d(val1, val2); }
int mul24(int val1, int val2) { return __nvvm_mul24_i(val1, val2); }
unsigned int mul24(unsigned int val1, unsigned int val2) {
  return __nvvm_mul24_ui(val1, val2);
}
float div_ftz(float val1, float val2) {
  return __nvvm_div_approx_ftz_f(val1, val2);
}
float div(float val1, float val2) { return __nvvm_div_approx_f(val1, val2); }
float div_rn_ftz(float val1, float val2) {
  return __nvvm_div_rn_ftz_f(val1, val2);
}
float div_rn(float val1, float val2) { return __nvvm_div_rn_f(val1, val2); }
float div_rz_ftz(float val1, float val2) {
  return __nvvm_div_rz_ftz_f(val1, val2);
}
float div_rz(float val1, float val2) { return __nvvm_div_rz_f(val1, val2); }
float div_rm_ftz(float val1, float val2) {
  return __nvvm_div_rm_ftz_f(val1, val2);
}
float div_rm(float val1, float val2) { return __nvvm_div_rm_f(val1, val2); }
float div_rp_ftz(float val1, float val2) {
  return __nvvm_div_rp_ftz_f(val1, val2);
}
float div_rp(float val1, float val2) { return __nvvm_div_rp_f(val1, val2); }
double div_rn(double val1, double val2) { return __nvvm_div_rn_d(val1, val2); }
double div_rz(double val1, double val2) { return __nvvm_div_rz_d(val1, val2); }
double div_rm(double val1, double val2) { return __nvvm_div_rm_d(val1, val2); }
double div_rp(double val1, double val2) { return __nvvm_div_rp_d(val1, val2); }
int brev32(int val) { return __nvvm_brev32(val); }
long brev64(long val) { return __nvvm_brev64(val); }
int sad(int val1, int val2, int val3) { return __nvvm_sad_i(val1, val2, val3); }
unsigned int sad(unsigned int val1, unsigned int val2, unsigned int val3) {
  return __nvvm_sad_ui(val1, val2, val3);
}
float floor_ftz(float val) { return __nvvm_floor_ftz_f(val); }
float floor(float val) { return __nvvm_floor_f(val); }
double floor(double val) { return __nvvm_floor_d(val); }
float ceil_ftz(float val) { return __nvvm_ceil_ftz_f(val); }
float ceil(float val) { return __nvvm_ceil_f(val); }
double ceil(double val) { return __nvvm_ceil_d(val); }
int abs(int val) { return __nvvm_abs_i(val); }
long long abs(long long val) { return __nvvm_abs_ll(val); }
float fabs_ftz(float val) { return __nvvm_fabs_ftz_f(val); }
float fabs(float val) { return __nvvm_fabs_f(val); }
double fabs(double val) { return __nvvm_fabs_d(val); }
float round_ftz(float val) { return __nvvm_round_ftz_f(val); }
float round(float val) { return __nvvm_round_f(val); }
double round(double val) { return __nvvm_round_d(val); }
float trunc_ftz(float val) { return __nvvm_trunc_ftz_f(val); }
float trunc(float val) { return __nvvm_trunc_f(val); }
double trunc(double val) { return __nvvm_trunc_d(val); }
float saturate_ftz(float val) { return __nvvm_saturate_ftz_f(val); }
float saturate(float val) { return __nvvm_saturate_f(val); }
double saturate(double val) { return __nvvm_saturate_d(val); }
float ex2_ftz(float val) { return __nvvm_ex2_approx_ftz_f(val); }
float ex2(float val) { return __nvvm_ex2_approx_f(val); }
double ex2(double val) { return __nvvm_ex2_approx_d(val); }
float lg2_ftz(float val) { return __nvvm_lg2_approx_ftz_f(val); }
float lg2(float val) { return __nvvm_lg2_approx_f(val); }
double lg2(double val) { return __nvvm_lg2_approx_d(val); }
float sin_ftz(float val) { return __nvvm_sin_approx_ftz_f(val); }
float sin(float val) { return __nvvm_sin_approx_f(val); }
float cos_ftz(float val) { return __nvvm_cos_approx_ftz_f(val); }
float cos(float val) { return __nvvm_cos_approx_f(val); }
float fma_rn_ftz(float val1, float val2, float val3) {
  return __nvvm_fma_rn_ftz_f(val1, val2, val3);
}
float fma_rn(float val1, float val2, float val3) {
  return __nvvm_fma_rn_f(val1, val2, val3);
}
float fma_rz_ftz(float val1, float val2, float val3) {
  return __nvvm_fma_rz_ftz_f(val1, val2, val3);
}
float fma_rz(float val1, float val2, float val3) {
  return __nvvm_fma_rz_f(val1, val2, val3);
}
float fma_rm_ftz(float val1, float val2, float val3) {
  return __nvvm_fma_rm_ftz_f(val1, val2, val3);
}
float fma_rm(float val1, float val2, float val3) {
  return __nvvm_fma_rm_f(val1, val2, val3);
}
float fma_rp_ftz(float val1, float val2, float val3) {
  return __nvvm_fma_rp_ftz_f(val1, val2, val3);
}
float fma_rp(float val1, float val2, float val3) {
  return __nvvm_fma_rp_f(val1, val2, val3);
}
double fma_rn(double val1, double val2, double val3) {
  return __nvvm_fma_rn_d(val1, val2, val3);
}
double fma_rz(double val1, double val2, double val3) {
  return __nvvm_fma_rz_d(val1, val2, val3);
}
double fma_rm(double val1, double val2, double val3) {
  return __nvvm_fma_rm_d(val1, val2, val3);
}
double fma_rp(double val1, double val2, double val3) {
  return __nvvm_fma_rp_d(val1, val2, val3);
}
float rcp_rn_ftz(float val) { return __nvvm_rcp_rn_ftz_f(val); }
float rcp_rn(float val) { return __nvvm_rcp_rn_f(val); }
float rcp_rz_ftz(float val) { return __nvvm_rcp_rz_ftz_f(val); }
float rcp_rz(float val) { return __nvvm_rcp_rz_f(val); }
float rcp_rm_ftz(float val) { return __nvvm_rcp_rm_ftz_f(val); }
float rcp_rm(float val) { return __nvvm_rcp_rm_f(val); }
float rcp_rp_ftz(float val) { return __nvvm_rcp_rp_ftz_f(val); }
float rcp_rp(float val) { return __nvvm_rcp_rp_f(val); }
double rcp_rn(double val) { return __nvvm_rcp_rn_d(val); }
double rcp_rz(double val) { return __nvvm_rcp_rz_d(val); }
double rcp_rm(double val) { return __nvvm_rcp_rm_d(val); }
double rcp_rp(double val) { return __nvvm_rcp_rp_d(val); }
double rcp_ftz(double val) { return __nvvm_rcp_approx_ftz_d(val); }
float sqrt_rn_ftz(float val) { return __nvvm_sqrt_rn_ftz_f(val); }
float sqrt_rn(float val) { return __nvvm_sqrt_rn_f(val); }
float sqrt_rz_ftz(float val) { return __nvvm_sqrt_rz_ftz_f(val); }
float sqrt_rz(float val) { return __nvvm_sqrt_rz_f(val); }
float sqrt_rm_ftz(float val) { return __nvvm_sqrt_rm_ftz_f(val); }
float sqrt_rm(float val) { return __nvvm_sqrt_rm_f(val); }
float sqrt_rp_ftz(float val) { return __nvvm_sqrt_rp_ftz_f(val); }
float sqrt_rp(float val) { return __nvvm_sqrt_rp_f(val); }
float sqrt_ftz(float val) { return __nvvm_sqrt_approx_ftz_f(val); }
float sqrt_ap(float val) { return __nvvm_sqrt_approx_f(val); }
double sqrt_rn(double val) { return __nvvm_sqrt_rn_d(val); }
double sqrt_rz(double val) { return __nvvm_sqrt_rz_d(val); }
double sqrt_rm(double val) { return __nvvm_sqrt_rm_d(val); }
double sqrt_rp(double val) { return __nvvm_sqrt_rp_d(val); }
float rsqrt_ftz(float val) { return __nvvm_rsqrt_approx_ftz_f(val); }
float rsqrt(float val) { return __nvvm_rsqrt_approx_f(val); }
double rsqrt(double val) { return __nvvm_rsqrt_approx_d(val); }
float add_rn_ftz(float val1, float val2) {
  return __nvvm_add_rn_ftz_f(val1, val2);
}
float add_rn(float val1, float val2) { return __nvvm_add_rn_f(val1, val2); }
float add_rz_ftz(float val1, float val2) {
  return __nvvm_add_rz_ftz_f(val1, val2);
}
float add_rz(float val1, float val2) { return __nvvm_add_rz_f(val1, val2); }
float add_rm_ftz(float val1, float val2) {
  return __nvvm_add_rm_ftz_f(val1, val2);
}
float add_rm(float val1, float val2) { return __nvvm_add_rm_f(val1, val2); }
float add_rp_ftz(float val1, float val2) {
  return __nvvm_add_rp_ftz_f(val1, val2);
}
float add_rp(float val1, float val2) { return __nvvm_add_rp_f(val1, val2); }
double add_rn(double val1, double val2) { return __nvvm_add_rn_d(val1, val2); }
double add_rz(double val1, double val2) { return __nvvm_add_rz_d(val1, val2); }
double add_rm(double val1, double val2) { return __nvvm_add_rm_d(val1, val2); }
double add_rp(double val1, double val2) { return __nvvm_add_rp_d(val1, val2); }
float d2f_rn_ftz(double val) { return __nvvm_d2f_rn_ftz(val); }
float d2f_rn(double val) { return __nvvm_d2f_rn(val); }
float d2f_rz_ftz(double val) { return __nvvm_d2f_rz_ftz(val); }
float d2f_rz(double val) { return __nvvm_d2f_rz(val); }
float d2f_rm_ftz(double val) { return __nvvm_d2f_rm_ftz(val); }
float d2f_rm(double val) { return __nvvm_d2f_rm(val); }
float d2f_rp_ftz(double val) { return __nvvm_d2f_rp_ftz(val); }
float d2f_rp(double val) { return __nvvm_d2f_rp(val); }
int d2i_rn(double val) { return __nvvm_d2i_rn(val); }
int d2i_rz(double val) { return __nvvm_d2i_rz(val); }
int d2i_rm(double val) { return __nvvm_d2i_rm(val); }
int d2i_rp(double val) { return __nvvm_d2i_rp(val); }
unsigned int d2ui_rn(double val) { return __nvvm_d2ui_rn(val); }
unsigned int d2ui_rz(double val) { return __nvvm_d2ui_rz(val); }
unsigned int d2ui_rm(double val) { return __nvvm_d2ui_rm(val); }
unsigned int d2ui_rp(double val) { return __nvvm_d2ui_rp(val); }
double i2d_rn(int val) { return __nvvm_i2d_rn(val); }
double i2d_rz(int val) { return __nvvm_i2d_rz(val); }
double i2d_rm(int val) { return __nvvm_i2d_rm(val); }
double i2d_rp(int val) { return __nvvm_i2d_rp(val); }
double ui2d_rn(unsigned int val) { return __nvvm_ui2d_rn(val); }
double ui2d_rz(unsigned int val) { return __nvvm_ui2d_rz(val); }
double ui2d_rm(unsigned int val) { return __nvvm_ui2d_rm(val); }
double ui2d_rp(unsigned int val) { return __nvvm_ui2d_rp(val); }
int f2i_rn_ftz(float val) { return __nvvm_f2i_rn_ftz(val); }
int f2i_rn(float val) { return __nvvm_f2i_rn(val); }
int f2i_rz_ftz(float val) { return __nvvm_f2i_rz_ftz(val); }
int f2i_rz(float val) { return __nvvm_f2i_rz(val); }
int f2i_rm_ftz(float val) { return __nvvm_f2i_rm_ftz(val); }
int f2i_rm(float val) { return __nvvm_f2i_rm(val); }
int f2i_rp_ftz(float val) { return __nvvm_f2i_rp_ftz(val); }
int f2i_rp(float val) { return __nvvm_f2i_rp(val); }
unsigned int f2ui_rn_ftz(float val) { return __nvvm_f2ui_rn_ftz(val); }
unsigned int f2ui_rn(float val) { return __nvvm_f2ui_rn(val); }
unsigned int f2ui_rz_ftz(float val) { return __nvvm_f2ui_rz_ftz(val); }
unsigned int f2ui_rz(float val) { return __nvvm_f2ui_rz(val); }
unsigned int f2ui_rm_ftz(float val) { return __nvvm_f2ui_rm_ftz(val); }
unsigned int f2ui_rm(float val) { return __nvvm_f2ui_rm(val); }
unsigned int f2ui_rp_ftz(float val) { return __nvvm_f2ui_rp_ftz(val); }
unsigned int f2ui_rp(float val) { return __nvvm_f2ui_rp(val); }
float i2f_rn(int val) { return __nvvm_i2f_rn(val); }
float i2f_rz(int val) { return __nvvm_i2f_rz(val); }
float i2f_rm(int val) { return __nvvm_i2f_rm(val); }
float i2f_rp(int val) { return __nvvm_i2f_rp(val); }
float ui2f_rn(unsigned int val) { return __nvvm_ui2f_rn(val); }
float ui2f_rz(unsigned int val) { return __nvvm_ui2f_rz(val); }
float ui2f_rm(unsigned int val) { return __nvvm_ui2f_rm(val); }
float ui2f_rp(unsigned int val) { return __nvvm_ui2f_rp(val); }
double lohi_i2d(int val1, int val2) { return __nvvm_lohi_i2d(val1, val2); }
int d2i_lo(double val) { return __nvvm_d2i_lo(val); }
int d2i_hi(double val) { return __nvvm_d2i_hi(val); }
long long f2ll_rn_ftz(float val) { return __nvvm_f2ll_rn_ftz(val); }
long long f2ll_rn(float val) { return __nvvm_f2ll_rn(val); }
long long f2ll_rz_ftz(float val) { return __nvvm_f2ll_rz_ftz(val); }
long long f2ll_rz(float val) { return __nvvm_f2ll_rz(val); }
long long f2ll_rm_ftz(float val) { return __nvvm_f2ll_rm_ftz(val); }
long long f2ll_rm(float val) { return __nvvm_f2ll_rm(val); }
long long f2ll_rp_ftz(float val) { return __nvvm_f2ll_rp_ftz(val); }
long long f2ll_rp(float val) { return __nvvm_f2ll_rp(val); }
unsigned long long f2ull_rn_ftz(float val) { return __nvvm_f2ull_rn_ftz(val); }
unsigned long long f2ull_rn(float val) { return __nvvm_f2ull_rn(val); }
unsigned long long f2ull_rz_ftz(float val) { return __nvvm_f2ull_rz_ftz(val); }
unsigned long long f2ull_rz(float val) { return __nvvm_f2ull_rz(val); }
unsigned long long f2ull_rm_ftz(float val) { return __nvvm_f2ull_rm_ftz(val); }
unsigned long long f2ull_rm(float val) { return __nvvm_f2ull_rm(val); }
unsigned long long f2ull_rp_ftz(float val) { return __nvvm_f2ull_rp_ftz(val); }
unsigned long long f2ull_rp(float val) { return __nvvm_f2ull_rp(val); }
long long d2ll_rn(double val) { return __nvvm_d2ll_rn(val); }
long long d2ll_rz(double val) { return __nvvm_d2ll_rz(val); }
long long d2ll_rm(double val) { return __nvvm_d2ll_rm(val); }
long long d2ll_rp(double val) { return __nvvm_d2ll_rp(val); }
unsigned long long d2ull_rn(double val) { return __nvvm_d2ull_rn(val); }
unsigned long long d2ull_rz(double val) { return __nvvm_d2ull_rz(val); }
unsigned long long d2ull_rm(double val) { return __nvvm_d2ull_rm(val); }
unsigned long long d2ull_rp(double val) { return __nvvm_d2ull_rp(val); }
float ll2f_rn(long long val) { return __nvvm_ll2f_rn(val); }
float ll2f_rz(long long val) { return __nvvm_ll2f_rz(val); }
float ll2f_rm(long long val) { return __nvvm_ll2f_rm(val); }
float ll2f_rp(long long val) { return __nvvm_ll2f_rp(val); }
float ull2f_rn(unsigned long long val) { return __nvvm_ull2f_rn(val); }
float ull2f_rz(unsigned long long val) { return __nvvm_ull2f_rz(val); }
float ull2f_rm(unsigned long long val) { return __nvvm_ull2f_rm(val); }
float ull2f_rp(unsigned long long val) { return __nvvm_ull2f_rp(val); }
double ll2d_rn(long long val) { return __nvvm_ll2d_rn(val); }
double ll2d_rz(long long val) { return __nvvm_ll2d_rz(val); }
double ll2d_rm(long long val) { return __nvvm_ll2d_rm(val); }
double ll2d_rp(long long val) { return __nvvm_ll2d_rp(val); }
double ull2d_rn(unsigned long long val) { return __nvvm_ull2d_rn(val); }
double ull2d_rz(unsigned long long val) { return __nvvm_ull2d_rz(val); }
double ull2d_rm(unsigned long long val) { return __nvvm_ull2d_rm(val); }
double ull2d_rp(unsigned long long val) { return __nvvm_ull2d_rp(val); }
short f2h_rn_ftz(float val) { return __nvvm_f2h_rn_ftz(val); }
short f2h_rn(float val) { return __nvvm_f2h_rn(val); }
float h2f(short val) { return __nvvm_h2f(val); }
int bitcast_f2i(float val) { return __nvvm_bitcast_f2i(val); }
float bitcast_i2f(int val) { return __nvvm_bitcast_i2f(val); }
double bitcast_ll2d(long long val) { return __nvvm_bitcast_ll2d(val); }
long long bitcast_d2ll(double val) { return __nvvm_bitcast_d2ll(val); }
}
*/
//float exp(float val) { return ::exp(val); }
//
//float log(float val) { return ::log(val); }
//
//float sqrt(float val) { return ::sqrt(val); }
//
//double rsqrt(double val) { return ::rsqrt(val); }
//float rsqrt(float val) { return ::rsqrtf(val); }
//
//float sin(float val) { return ::sin(val); }
//
//float cos(float val) { return ::cos(val); }
//float fabs(float val) { return ::fabs(val); }

// barriers

void barrier() {
  ::barrier(1);
#ifdef __CUDA_DEVICE_CODE
  __asm__ volatile("" : : : "memory");
#endif
}

//
//namespace atomic {
//namespace global {
//int add(int *ptr, int value) {
//  return ::atomic_add((global_mem_int_ptr)ptr, value);
//}
//
//unsigned int add(unsigned int *ptr, unsigned int value) {
//  return ::atomic_add((global_mem_uint_ptr)ptr, value);
//}
//}
//
//namespace shared {
//int add(int *ptr, int value) {
//  return ::atomic_add((shared_mem_int_ptr)ptr, value);
//}
//
//unsigned int add(unsigned int *ptr, unsigned int value) {
//  return ::atomic_add((shared_mem_uint_ptr)ptr, value);
//}
//}
//}
}

#else

#include <cmath>

namespace native {

  namespace index {
    template <> unsigned int x<idx::thread>() { return 0; }
    template <> unsigned int y<idx::thread>() { return 0; }
    template <> unsigned int z<idx::thread>() { return 0; }

    template <> unsigned int x<dim::block>() { return 1; }
    template <> unsigned int y<dim::block>() { return 1; }
    template <> unsigned int z<dim::block>() { return 1; }

    template <> unsigned int x<idx::block>() { return 0; }
    template <> unsigned int y<idx::block>() { return 0; }
    template <> unsigned int z<idx::block>() { return 0; }

    template <> unsigned int x<dim::grid>() { return 1; }
    template <> unsigned int y<dim::grid>() { return 1; }
    template <> unsigned int z<dim::grid>() { return 1; }

    template <> unsigned int x<idx::global>() { return 0; }
    template <> unsigned int y<idx::global>() { return 0; }
    template <> unsigned int z<idx::global>() { return 0; }
  }

  float sin(float val) { return std::sin(val); }

  float fabs(float val) { return std::abs(val); }

  float exp(float val) { return std::exp(val); }

  float log(float val) { return std::log(val); }

  float sqrt(float val) { return std::sqrt(val); }

  extern "C" float __log2f(float val){ return std::log2(val); }
  extern "C" float __powf(float val, float e) { return std::pow(val, e); }

  void barrier(){/* noop */};
}

unsigned get_global_id(unsigned) { return 0; }

#endif

Thread Thread::get() {
  using namespace native::index;
  return {
      {static_cast<int>(x<idx::thread>()), static_cast<int>(y<idx::thread>()),
          static_cast<int>(z<idx::thread>())},
      {static_cast<int>(x<idx::global>()), static_cast<int>(y<idx::global>()),
          static_cast<int>(z<idx::global>())}};
}

Block Block::get() {
  using namespace native::index;
  return {{static_cast<int>(x<idx::block>()), static_cast<int>(y<idx::block>()),
              static_cast<int>(z<idx::block>())},
          {static_cast<int>(x<dim::block>()), static_cast<int>(y<dim::block>()),
              static_cast<int>(z<dim::block>())}};
}

void Block::synchronize() { native::barrier(); }

Grid Grid::get() {
  using namespace native::index;
  return {{static_cast<int>(x<dim::grid>()), static_cast<int>(y<dim::grid>()),
              static_cast<int>(z<dim::grid>())}};
}
