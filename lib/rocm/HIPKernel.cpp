//===-----------------------------------------------------------*- C++ -*-===//
//
//                       The LLVM-based PACXX Project
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "pacxx/detail/rocm/HIPKernel.h"
#include "pacxx/detail/common/Log.h"
#include "pacxx/detail/rocm/HIPErrorDetection.h"
#include "pacxx/detail/rocm/HIPRuntime.h"
#include <hip/hip_runtime.h>

namespace pacxx {
namespace v2 {

HIPKernel::HIPKernel(HIPRuntime &runtime, hipFunction_t fptr, std::string name)
    : Kernel(runtime, name), _runtime(runtime), _fptr(fptr) {}

HIPKernel::~HIPKernel() {}

void HIPKernel::configurate(KernelConfiguration config) {
  if (_config != config) {
    _config = config;
    std::vector<size_t> a(6);
    a[0] = config.threads.x;
    a[1] = config.threads.y;
    a[2] = config.threads.z;
    a[3] = config.blocks.x;
    a[4] = config.blocks.y;
    a[5] = config.blocks.z;

    for (size_t i = 0; i < a.size(); ++i) {
      // setStagedValue((i * -1) - 1, a[i], true);
    }
  }
}

void HIPKernel::launch() {
  if (!_fptr || _staged_values_changed) { // kernel has no function ptr yet.
                                          // request kernel transformation and
                                          // recompilation if necessary
    _runtime.requestIRTransformation(*this);
    _staged_values_changed = false;
  }

  __verbose("setting kernel arguments");
  _launch_args.clear(); // remove old args first
  _launch_args.push_back(HIP_LAUNCH_PARAM_BUFFER_POINTER);
  _launch_args.push_back(const_cast<void*>(_lambdaPtr));
  _launch_args.push_back(HIP_LAUNCH_PARAM_BUFFER_SIZE);
  _launch_args.push_back(reinterpret_cast<void *>(&_argBufferSize));
  _launch_args.push_back(HIP_LAUNCH_PARAM_END);

  __debug("Launching kernel: ", _name);
  __verbose("Kernel configuration: \nblocks(", _config.blocks.x, ",",
            _config.blocks.y, ",", _config.blocks.z, ")\nthreads(",
            _config.threads.x, ",", _config.threads.y, ",", _config.threads.z,
            ")\nshared_mem=", _config.sm_size);

  SEC_HIP_CALL(hipModuleLaunchKernel(
      _fptr, _config.blocks.x, _config.blocks.y, _config.blocks.z,
      _config.threads.x, _config.threads.y, _config.threads.z, _config.sm_size,
      NULL, nullptr, &_launch_args[0]));
 // if (_callback)  FIXME: add callback support in HIP backend
 //   SEC_HIP_CALL(cudaStreamAddCallback(nullptr, HIPRuntime::fireCallback,
 //                                       &_callback, 0));
}


}
}
