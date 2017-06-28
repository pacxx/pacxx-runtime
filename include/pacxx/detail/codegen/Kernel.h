//
// Created by m_haid02 on 19.05.17.
//

#pragma once

#include "pacxx/detail/KernelConfiguration.h"
#include "pacxx/detail/common/Log.h"
#include "pacxx/detail/common/Meta.h"
#include "pacxx/detail/device/DeviceFunctionDecls.h"
#include <cstddef>
#include <regex>
#include <type_traits>
#include <utility>
#include <type_traits>
#include <experimental/type_traits>

namespace pacxx {
namespace v2 {
enum class Target {
  Generic,
  GPU,
  CPU
};

class Executor;

//
// THIS IS THE GENERIC KERNEL EXPRESSION
//

// forward decl the kernel body
template<typename L>
auto kernelBody(L &&callable);

struct range {
private:
  // a range object should never be copied or moved and only be handled as reference
  range() = default;
  template<typename L> friend auto kernelBody(L &&callable);

public:
  range(const range &) = delete;
  range(range &&) = delete;
  auto operator=(const range &) = delete;
  auto operator=(range &&) = delete;

  auto get_global(unsigned int dim) { return ::get_global_id(dim); }
  auto get_local(unsigned int dim) { return ::get_local_id(dim); }
  auto get_block(unsigned int dim) { return ::get_group_id(dim); }
  auto get_block_size(unsigned int dim) { return ::get_local_size(dim); }
  auto get_num_blocks(unsigned int dim) { return ::get_num_groups(dim); }
  auto get_grid_size(unsigned int dim) { return ::get_grid_size(dim); }
  auto synchronize() { barrier(0); }
};

template<typename L>
auto kernelBody(L &&callable) {
#ifdef __device_code__
  pacxx::v2::range thread;
  callable(thread);
#endif
}



template<typename L>
[[pacxx::kernel]] [[pacxx::target("Generic")]] void genericKernel(L callable) noexcept {
  kernelBody(callable);
}

template<typename L>
[[pacxx::kernel]] [[pacxx::target("GPU")]] void genericKernelGPU(L callable) noexcept {
  kernelBody(callable);
}

template<typename L>
[[pacxx::kernel]] [[pacxx::target("CPU")]] void genericKernelCPU(L callable) noexcept {
  kernelBody(callable);
}


template<typename L, Target targ> class _kernel {
public:
  friend class Executor;

  _kernel(L lambda)
      : _function(std::forward<L>(lambda)) {
    switch (targ) {
    case Target::Generic:genericKernel(_function);
      break;
    case Target::CPU:genericKernelCPU(_function);
      break;
    case Target::GPU:genericKernelGPU(_function);
      break;
    }
  }

  L _function;
};


template<typename Func, Target targ = Target::Generic>
auto codegenKernel(Func &lambda) { return _kernel<Func, targ>(std::forward<Func>(lambda)); }

}
}