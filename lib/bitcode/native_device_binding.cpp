#include "pacxx/detail/device/DeviceCode.h"
#include "pacxx/detail/device/DeviceFunctionDecls.h"
#include <math.h>

__forceinline__ __index_t get_global_id(unsigned int dimindx) {
  switch (dimindx) {
  case 0:
    return __pacxx_read_ntid_x() *
           __pacxx_read_ctaid_x() +
           __pacxx_read_tid_x();
  case 1:
    return __pacxx_read_ntid_y() *
           __pacxx_read_ctaid_y() +
           __pacxx_read_tid_y();
  case 2:
    return __pacxx_read_ntid_z() *
           __pacxx_read_ctaid_z() +
           __pacxx_read_tid_z();
  default:return 0;
  }
}

__forceinline__ __index_t get_local_id(unsigned int dimindx) {
  switch (dimindx) {
  case 0:return __pacxx_read_tid_x();
  case 1:return __pacxx_read_tid_y();
  case 2:return __pacxx_read_tid_z();
  default:return 0;
  }
}

__forceinline__ __index_t get_group_id(unsigned int dimindx) {
  switch (dimindx) {
  case 0:return __pacxx_read_ctaid_x();
  case 1:return __pacxx_read_ctaid_y();
  case 2:return __pacxx_read_ctaid_z();
  default:return 0;
  }
}

__forceinline__ __index_t get_local_size(unsigned int dimindx) {
  switch (dimindx) {
  case 0:return __pacxx_read_ntid_x();
  case 1:return __pacxx_read_ntid_y();
  case 2:return __pacxx_read_ntid_z();
  default:return 0;
  }
}

__forceinline__ __index_t get_num_groups(unsigned int dimindx) {
  switch (dimindx) {
  case 0:return __pacxx_read_nctaid_x();
  case 1:return __pacxx_read_nctaid_y();
  case 2:return __pacxx_read_nctaid_z();
  default:return 0;
  }
}

__forceinline__ __index_t get_grid_size(unsigned int dimindx) {
  switch (dimindx) {
  case 0:return __pacxx_read_ntid_x() * __pacxx_read_nctaid_x();
  case 1:return __pacxx_read_ntid_y() * __pacxx_read_nctaid_y();
  case 2:return __pacxx_read_ntid_z() * __pacxx_read_nctaid_z();
  default:return 0;
  }
}

__forceinline__ void barrier(unsigned int) { __pacxx_barrier(); }

extern "C" __forceinline__ double rsqrt(double val) {
  return 1.0 / sqrt(val);
}

extern "C" __forceinline__ float rsqrtf(float val) {
  return 1.0f / sqrtf(val);
}