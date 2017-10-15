//
// Created by mhaidl on 05/06/16.
//


//#include <type_traits>
#include "pacxx/detail/device/DeviceCode.h"
#include "pacxx/detail/device/DeviceFunctionDecls.h"

#ifdef __device_code__

///////////////////////////// INDEXING ////////////////////////////

using intptr_t = long;

/*typedef const __attribute__((address_space(4))) char *_format_t;
__forceinline__ void __vprintf_conv(_format_t ptr) {
  const char *out;
  asm("cvta.const.u64  %0, %1;" : "=r"(out) : "r"(ptr) :);
}
*/
///////////////////////////// ATOMICS ////////////////////////////
/*
#define __to_string(v) #v
#define __expand(v) __to_string(v)
#define __ptx_atom(op, memory, type)                                           \
  "{ \n\tatom." __expand(memory) "." __expand(op) "." __expand(                \
      type) " %0, [%1], %2; \n\t}"
#define __atomic_op(op, ctype, ptxtype, memory)                                \
  __forceinline__ ctype atomic_##op(memory##_mem_##ctype##_ptr ptr,            \
                                    ctype value) {                             \
    ctype old;                                                                 \
    intptr_t addr = reinterpret_cast<intptr_t>(ptr);                           \
    if (!std::is_floating_point<ctype>::value) {                               \
      if (sizeof(ctype) == 8)                                                  \
        asm volatile(__ptx_atom(op, memory, ptxtype)                           \
                     : "=l"(old)                                               \
                     : "l"(addr), "l"(value));                                 \
      else                                                                     \
        asm volatile(__ptx_atom(op, memory, ptxtype)                           \
                     : "=r"(old)                                               \
                     : "l"(addr), "r"(value));                                 \
    } else {                                                                   \
      asm volatile(__ptx_atom(op, memory, ptxtype)                             \
                   : "=f"(old)                                                 \
                   : "l"(addr), "f"(value));                                   \
    }                                                                          \
    return old;                                                                \
  }

#define atomic_(op, ctype, ptxtype)                                            \
  __atomic_op(op, ctype, ptxtype, global)                                      \
      __atomic_op(op, ctype, ptxtype, shared)

atomic_(add, int, s32);
atomic_(add, uint, u32);
atomic_(add, ulong, u64);
atomic_(add, float, f32);

atomic_(min, int, s32);
atomic_(min, uint, u32);
atomic_(min, long, s64);

atomic_(min, ulong, u64);

atomic_(max, int, s32);
atomic_(max, uint, u32);

atomic_(max, long, s64);
atomic_(max, ulong, u64);

atomic_(inc, uint, u32);
atomic_(dec, uint, u32);

atomic_(cas, uint, b32);
atomic_(cas, ulong, b64);

atomic_(exch, uint, b32);
atomic_(exch, ulong, b64);

atomic_(and, uint, b32);
atomic_(and, ulong, b64);

atomic_(or, uint, b32);

atomic_(or, ulong, b64);

atomic_ (xor, uint, b32);

atomic_ (xor, ulong, b64);

extern "C" {
  float __fAtomicAdd(float* ptr, float v) {
    return atomic_add((global_mem_float_ptr)ptr, v);
  }
}

*/

//__forceinline__ int atomic_add(global_mem_int_ptr ptr, int
// value)
//{
//    int old;
//    size_t addr = (size_t)ptr;
//    // asm("template-string" : "constraint"(output) :
//    "constraint"(input));
//	asm volatile(R"(
//{
//	atom.global.add.s32	%0, [%1], %2;
//}
//)"
//                 : "=r"(old)
//                 : "l"(addr), "r"(value));
//    return old;
//}
//
//__forceinline__ int atomic_add(shared_mem_int_ptr ptr, int
// value)
//{
//    int old;
//    size_t addr = (size_t)ptr;
//    // asm("template-string" : "constraint"(output) :
//    "constraint"(input));
//	asm volatile(R"(
//{
//	atom.shared.add.s32	%0, [%1], %2;
//}
//)"
//                 : "=r"(old)
//                 : "l"(addr), "r"(value));
//    return old;
//}
//
//__forceinline__ unsigned int atomic_add(global_mem_uint_ptr
// ptr, unsigned int
// value)
//{
//    unsigned int old;
//    size_t addr = (size_t)ptr;
//    // asm("template-string" : "constraint"(output) :
//    "constraint"(input));
//	asm volatile(R"(
//{
//	atom.global.add.u32	%0, [%1], %2;
//}
//)"
//                 : "=r"(old)
//                 : "l"(addr), "r"(value));
//    return old;
//}
//
//__forceinline__ unsigned int atomic_add(shared_mem_uint_ptr
// ptr, unsigned int
// value)
//{
//    unsigned int old;
//    size_t addr = (size_t)ptr;
//    // asm("template-string" : "constraint"(output) :
//    "constraint"(input));
// asm volatile(R"(
//{
//	atom.shared.add.u32	%0, [%1], %2;
//}
//)"
//                 : "=r"(old)
//                 : "l"(addr), "r"(value));
//    return old;
//}
//
//__forceinline__ long atomic_add(global_mem_long_ptr ptr,
// long value)
//{
//	long old;
//	size_t addr = (size_t)ptr;
//	// asm("template-string" : "constraint"(output) : "constraint"(input));
//	asm volatile(R"(
//{
//	atom.global.add.s64	%0, [%1], %2;
//}
//)"
//	: "=r"(old)
//		: "l"(addr), "l"(value));
//	return old;
//}
//
//__forceinline__ long atomic_add(shared_mem_long_ptr ptr,
// unsigned int value)
//{
//	unsigned int old;
//	size_t addr = (size_t)ptr;
//	// asm("template-string" : "constraint"(output) : "constraint"(input));
//	asm volatile(R"(
//{
//	atom.shared.add.s64	%0, [%1], %2;
//}
//)"
//	: "=r"(old)
//		: "l"(addr), "l"(value));
//	return old;
//}
//
//

///////////////////////////// MATH ////////////////////////////

extern "C" __forceinline__ int __mul24(int x, int y)
{
  return __nv_mul24(x, y);
}

extern "C" __forceinline__ float fmax(float v1, float v2) {
  return __nv_fmaxf(v1, v2);
}
extern "C" __forceinline__ float fmin(float v1, float v2) {
  return __nv_fminf(v1, v2);
}

extern "C" __forceinline__ float fmaxf(float v1, float v2) {
  return __nv_fmaxf(v1, v2);
}
extern "C" __forceinline__ float fminf(float v1, float v2) {
  return __nv_fminf(v1, v2);
}
extern "C" __forceinline__ double fabs(double val) {
  return __nv_fabs(val);
}
extern "C" __forceinline__ float fabsf(float val) {
  return __nv_fabsf(val);
}
extern "C" __forceinline__ double sqrt(double val) {
  return __nv_sqrt(val);
}
extern "C" __forceinline__ float sqrtf(float val) {
  return __nv_sqrtf(val);
}
extern "C" __forceinline__ double rsqrt(double val) {
  return __nv_rsqrt(val);
}
extern "C" __forceinline__ float rsqrtf(float val) {
  return __nv_rsqrtf(val);
}
extern "C" __forceinline__ double log(double val) {
  return __nv_log(val);
}
extern "C" __forceinline__ double exp(double val) {
  return __nv_exp(val);
}
extern "C" __forceinline__ double sin(double val) {
  return __nv_sin(val);
}
extern "C" __forceinline__ double cos(double val) {
  return __nv_cos(val);
}
extern "C" __forceinline__ float logf(float val) {
  return __nv_logf(val);
}
extern "C" __forceinline__ float expf(float val) {
  return __nv_expf(val);
}
extern "C" __forceinline__ float sinf(float val) {
  return __nv_sinf(val);
}
extern "C" __forceinline__ float cosf(float val) {
  return __nv_cosf(val);
}
extern "C" __forceinline__ float ceilf(float val) {
  return __nv_ceilf(val);
}
extern "C" __forceinline__ double ceil(double val) {
  return __nv_ceilf(val);
}
extern "C" __forceinline__ float __log2f(float a) { return __nv_fast_log2f(a); }
extern "C" __forceinline__ float __powf(float a, float b) {
  return __nv_fast_powf(a, b);
}
extern "C" __forceinline__ float log10f(float a) { return __nv_log10f(a); }
extern "C" __forceinline__ float exp10f(float a) { return __nv_exp10f(a); }
extern "C" __forceinline__ double log10(double a) { return __nv_log10(a); }
extern "C" __forceinline__ double exp10(double a) { return __nv_exp10(a); }


#endif

