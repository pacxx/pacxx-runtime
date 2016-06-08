//
// Created by mhaidl on 31/05/16.
//
#include "detail/cuda/CUDAKernel.h"
#include <cuda.h>
#include <detail/cuda/CUDAErrorDetection.h>
#include <detail/common/Log.h>

namespace pacxx {
namespace v2 {

CUDAKernel::CUDAKernel(CUfunction fptr) : _fptr(fptr) {}
CUDAKernel::~CUDAKernel() {}

void CUDAKernel::configurate(KernelConfiguration config) { _config = config; }

void CUDAKernel::setArguments(const std::vector<char> &arg_buffer) {
  _args = arg_buffer;
  _args_size = _args.size();
  _launch_args.clear(); // remove old args first
  _launch_args.push_back(CU_LAUNCH_PARAM_BUFFER_POINTER);
  _launch_args.push_back(reinterpret_cast<void *>(_args.data()));
  _launch_args.push_back(CU_LAUNCH_PARAM_BUFFER_SIZE);
  _launch_args.push_back(reinterpret_cast<void *>(&_args_size));
  _launch_args.push_back(CU_LAUNCH_PARAM_END);
}

void CUDAKernel::launch() {

  __verbose("Launching kernel: \nblocks(", _config.blocks.x, ",",
            _config.blocks.y, ",", _config.blocks.z, ")\nthreads(",
            _config.threads.x, ",", _config.threads.y, ",", _config.threads.z,
            ")\nshared_mem=", _config.sm_size);

  SEC_CUDA_CALL(cuLaunchKernel(
      _fptr, _config.blocks.x, _config.blocks.y, _config.blocks.z,
      _config.threads.x, _config.threads.y, _config.threads.z, _config.sm_size,
      NULL, nullptr, &_launch_args[0]));
}
}
}