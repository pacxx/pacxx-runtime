//
// Created by mhaidl on 05/06/16.
//

#ifndef PACXX_V2_DEVICETYPES_H
#define PACXX_V2_DEVICETYPES_H
#include <type_traits>

using size_t = std::conditional_t<sizeof(void *) == 4, unsigned int, unsigned long>;

using ulong = unsigned long;
using uint = unsigned int;

typedef volatile __attribute__((address_space(1))) int *global_mem_int_ptr;
typedef volatile
    __attribute__((address_space(1))) unsigned int *global_mem_uint_ptr;
typedef volatile __attribute__((address_space(1))) long *global_mem_long_ptr;
typedef volatile
    __attribute__((address_space(1))) unsigned long *global_mem_ulong_ptr;
typedef volatile __attribute__((address_space(1))) float *global_mem_float_ptr;

typedef volatile __attribute__((address_space(3))) int *shared_mem_int_ptr;
typedef volatile
    __attribute__((address_space(3))) unsigned int *shared_mem_uint_ptr;
typedef volatile __attribute__((address_space(3))) long *shared_mem_long_ptr;
typedef volatile
    __attribute__((address_space(3))) unsigned long *shared_mem_ulong_ptr;
typedef volatile __attribute__((address_space(3))) float *shared_mem_float_ptr;

struct _idx {

  _idx(int x, int y, int z) : x(x), y(y), z(z) {}

  int x;
  int y;
  int z;
};

struct Block {
  Block(_idx i, _idx r) : index(i), range(r) {}
  _idx index;
  _idx range;

  void synchronize();
  static Block get();
};

struct Thread {
  Thread(_idx i, _idx g) : index(i), global(g) {}
  _idx index;
  _idx global;
  static Thread get();
};

struct Grid {
  Grid(_idx r) : range(r) {}
  _idx range;
  static Grid get();
};

#endif // PACXX_V2_DEVICETYPES_H
