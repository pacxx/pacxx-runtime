//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "pacxx/detail/native/NativeKernel.h"
#include "pacxx/detail/common/Exceptions.h"
#include "pacxx/detail/common/Log.h"
#include "pacxx/detail/common/Timing.h"
#include "pacxx/detail/native/NativeRuntime.h"
#include "pacxx/detail/native/PAPIProfiler.h"
#include <llvm/IR/Module.h>

#ifndef PACXX_DISABLE_TBB
#include <tbb/tbb.h>
#else
#include <omp.h>
#endif

#include <fstream>

namespace pacxx {
namespace v2 {

NativeKernel::NativeKernel(NativeRuntime &runtime, void *fptr, std::string name)
    : Kernel(runtime, name), _runtime(runtime), _fptr(fptr), _runs(1) {
  auto runs = common::GetEnv("PACXX_NATIVE_KERNEL_RUNS");
  if (runs != "")
    _runs = std::stoul(runs);
}

NativeKernel::~NativeKernel() {}

void NativeKernel::configurate(KernelConfiguration config) {
  if (_config != config) {
    _config = config;
  }
}

NativeRuntime &NativeKernel::getRuntime() {
	return _runtime;
}

void NativeKernel::launch() {

  if (!_fptr || _staged_values_changed) { // kernel has no function ptr yet.
                                          // request kernel transformation and
                                          // recompilation if necessary
    _runtime.requestIRTransformation(*this);
    _staged_values_changed = false;
  }

  __verbose("Launching kernel: \nblocks(", _config.blocks.x, ",",
            _config.blocks.y, ",", _config.blocks.z, ")\nthreads(",
            _config.threads.x, ",", _config.threads.y, ",", _config.threads.z,
            ")\nshared_mem=", _config.sm_size);

  // The kernel wrapper always has this function signature.
  // The kernel args are constructed from the char buffer
  auto functor = reinterpret_cast<void (*)(int, int, int, int, int, int, int,
                                           int, int, int, const void *)>(_fptr);

  std::chrono::high_resolution_clock::time_point start, end;

  std::vector<unsigned> times(_runs);

#ifdef __PACXX_OMP
  for(unsigned i = 0; i < _runs; ++i) {
    start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for collapse(3)
    for(unsigned bidz = 0; bidz < _config.blocks.z; ++bidz)
      for(unsigned bidy = 0; bidy < _config.blocks.y; ++bidy)
        for(unsigned bidx = 0; bidx < _config.blocks.x; ++bidx)
          functor(bidx, bidy, bidz, _config.blocks.x, _config.blocks.y,
                  _config.blocks.z, _config.threads.x, _config.threads.y,
                  _config.threads.z, _config.sm_size, _lambdaPtr);

    end = std::chrono::high_resolution_clock::now();
    times[i] = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  }
#else
  for (unsigned i = 0; i < _runs; ++i) {
    start = std::chrono::high_resolution_clock::now();
    tbb::parallel_for(size_t(0), _config.blocks.z, [&](size_t bidz) {
      tbb::parallel_for(size_t(0), _config.blocks.y, [&](size_t bidy) {
        tbb::parallel_for(size_t(0), _config.blocks.x, [&](size_t bidx) {
          functor(bidx, bidy, bidz, _config.blocks.x, _config.blocks.y,
                  _config.blocks.z, _config.threads.x, _config.threads.y,
                  _config.threads.z, _config.sm_size, _lambdaPtr);
        });
      });
    });
    end = std::chrono::high_resolution_clock::now();
    times[i] = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  }
#endif

  auto first = times.begin() + (_runs == 1 ? 0 : 1); // ignore the first run as warmup
  auto last = times.end();

  auto med_v = median(first, last);
  auto min_v = *std::min_element(first, last);
  auto max_v = *std::max_element(first, last);
  auto avg_v = average(first, last);
  auto dev_v = deviation(first, last);

  __verbose("Time measured in runtime : ",
            med_v,
            " (",
            min_v,
            " ",
            max_v,
            " ",
            avg_v,
            " ",
            dev_v,
            ") us (",
            (last - first),
            " iterations)");
  std::ofstream f(std::string(program_invocation_name) + "-timing");
  std::ostream_iterator<unsigned> output_iterator(f, "\n");
  std::copy(times.begin(), times.end(), output_iterator);

  if (_callback)
    _callback();
}

void NativeKernel::profile() {
  PAPIProfiler* ptr = static_cast<PAPIProfiler*>(_runtime.getProfiler());
  ptr->updateKernel(this);
  ptr->dryrun();
  ptr->profile();
}


}
}
