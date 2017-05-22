//
// Created by mhaidl on 31/05/16.
//
#include "pacxx/detail/cuda/CUDAKernel.h"
#include "pacxx/detail/common/Log.h"
#include "pacxx/detail/cuda/CUDAErrorDetection.h"
#include "pacxx/detail/cuda/CUDARuntime.h"
#include <cuda.h>

namespace pacxx {
namespace v2 {

CUDAKernel::CUDAKernel(CUDARuntime &runtime, CUfunction fptr, std::string name)
    : Kernel(runtime, name), _runtime(runtime), _fptr(fptr) {}

CUDAKernel::~CUDAKernel() {}

void CUDAKernel::configurate(KernelConfiguration config) {
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

void CUDAKernel::launch() {
  if (!_fptr || _staged_values_changed) { // kernel has no function ptr yet.
                                          // request kernel transformation and
                                          // recompilation if necessary
    _runtime.requestIRTransformation(*this);
    _staged_values_changed = false;
  }

  __verbose("setting kernel arguments");
  _launch_args.clear(); // remove old args first
  _launch_args.push_back(CU_LAUNCH_PARAM_BUFFER_POINTER);
  _launch_args.push_back((void *) (_lambdaPtr));
  _launch_args.push_back(CU_LAUNCH_PARAM_BUFFER_SIZE);
  _launch_args.push_back(reinterpret_cast<void *>(&_argBufferSize));
  _launch_args.push_back(CU_LAUNCH_PARAM_END);

  __verbose("Launching kernel: \nblocks(", _config.blocks.x, ",",
            _config.blocks.y, ",", _config.blocks.z, ")\nthreads(",
            _config.threads.x, ",", _config.threads.y, ",", _config.threads.z,
            ")\nshared_mem=", _config.sm_size);

  SEC_CUDA_CALL(cuLaunchKernel(
      _fptr, _config.blocks.x, _config.blocks.y, _config.blocks.z,
      _config.threads.x, _config.threads.y, _config.threads.z, _config.sm_size,
      NULL, nullptr, &_launch_args[0]));
  if (_callback)
    SEC_CUDA_CALL(cudaStreamAddCallback(nullptr, CUDARuntime::fireCallback,
                                        &_callback, 0));
}


}
}